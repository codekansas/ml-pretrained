"""Defines a simple API for an audio quantizer model that runs on Mels.

.. highlight:: python
.. code-block:: python

    from pretrained.mel_codec import pretrained_mel_codec

    model = pretrained_mel_codec("librivox")
    quantizer, dequantizer = model.quantizer(), model.dequantizer()

    # Convert some audio to a quantized representation.
    quantized = quantizer(audio)

    # Convert the quantized representation back to audio.
    audio = dequantizer(quantized)
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Literal, cast, get_args

import safetensors.torch
import torch
import torchaudio
from ml.models.codebook import ResidualVectorQuantization, VectorQuantization
from ml.models.embeddings import get_positional_embeddings
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer
from torch import Tensor, nn

from pretrained.vocoder.hifigan import AudioToHifiGanMels, pretrained_hifigan

logger = logging.getLogger(__name__)

PretrainedMelCodecType = Literal["librivox"]


def cast_pretrained_mel_codec_type(s: str) -> PretrainedMelCodecType:
    if s not in get_args(PretrainedMelCodecType):
        raise KeyError(f"Invalid Codec type: {s} Expected one of: {get_args(PretrainedMelCodecType)}")
    return cast(PretrainedMelCodecType, s)


@dataclass
class MelCodecConfig:
    num_mels: int
    d_model: int
    nhead: int
    dim_feedforward: int
    num_layers: int
    codebook_size: int
    num_quantizers: int
    encoder_causal: bool
    max_tsz: int


class MelCodec(nn.Module):
    """Defines an audio transformer module.

    This module takes the Mel spectrogram as an input and outputs the
    predicted next step of the Mel spectrogram.

    Parameters:
        num_mels: The number of Mel spectrogram channels.
        d_model: The dimensionality of the transformer model.
        nhead: The number of transformer attention heads.
        dim_feedforward: The dimensionality of the feedforward network.
        num_layers: The number of layers in the encoder and decoder.
        codebook_size: The size of the codebook.
        num_quantizers: The number of quantizers in the residual vector
            quantization.
        encoder_causal: Whether to use a causal encoder.
        max_tsz: The maximum time dimension size.
    """

    __constants__ = ["codebook_size", "num_mels", "encoder_causal"]

    def __init__(
        self,
        num_mels: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        codebook_size: int,
        num_quantizers: int,
        encoder_causal: bool = False,
        max_tsz: int = 2048,
    ) -> None:
        super().__init__()

        self.codebook_size = codebook_size
        self.num_mels = num_mels
        self.encoder_causal = encoder_causal

        self.rvq = ResidualVectorQuantization(
            VectorQuantization(
                dim=d_model,
                codebook_size=codebook_size,
                kmeans_init=False,
            ),
            num_quantizers=num_quantizers,
        )

        self.embs = get_positional_embeddings(max_tsz, d_model, "rotary")
        self.init_emb = nn.Parameter(torch.zeros(1, 1, d_model))

        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(max_tsz), persistent=False)

        self.encoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )

        self.decoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )

        self.enc_in_proj = nn.Linear(num_mels, d_model)
        self.dec_in_proj = nn.Linear(num_mels, d_model)
        self.out_proj = nn.Linear(d_model, num_mels)

    mask: Tensor

    def _featurize(self, x: Tensor) -> Tensor:
        x = self.enc_in_proj(x)
        x = self.embs(x)
        if self.encoder_causal:
            x = self.encoder_transformer(x, mask=self.mask, is_causal=True)
        else:
            x = self.encoder_transformer(x, is_causal=False)
        return x

    def _train_from_codes(self, x_codes: Tensor, x_prev: Tensor) -> Tensor:
        x_prev = self.dec_in_proj(x_prev[:, :-1])
        x_prev = torch.cat([self.init_emb.repeat(x_prev.shape[0], 1, 1), x_prev], dim=1)
        x = x_codes + x_prev
        x = self.decoder_transformer(x, mask=self.mask, is_causal=True)
        x = self.out_proj(x)
        return x

    def _infer_from_codes(self, x_codes: Tensor) -> Tensor:
        init_emb = self.init_emb.repeat(x_codes.shape[0], 1, 1)
        x_nb = init_emb
        tsz = x_codes.shape[1]
        for t in range(tsz):
            x = x_codes[:, : x_nb.shape[1]] + x_nb
            x = self.decoder_transformer(x, mask=self.mask, is_causal=True)
            x = self.out_proj(x)
            if t < tsz - 1:
                x_nb = torch.cat([init_emb, self.dec_in_proj(x)], dim=1)
        return x

    def _pre_quant_to_tokens(self, xq: Tensor) -> Tensor:
        return self.rvq.encode(xq.transpose(1, 2))

    def _tokens_to_embedding(self, tokens: Tensor) -> Tensor:
        return self.rvq.decode(tokens).transpose(1, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Runs the forward pass of the model.

        Args:
            x: The input Mel spectrogram, with shape ``(B, T, C)``.

        Returns:
            The predicted next step of the Mel spectrogram, with shape
            ``(B, T, C)``, along with the codebook loss.
        """
        xq = self._featurize(x).transpose(1, 2)
        xq, _, codebook_loss, _ = self.rvq(xq)
        x = self._train_from_codes(xq.transpose(1, 2), x)
        return x, codebook_loss

    def infer(self, x: Tensor) -> Tensor:
        xq = self._featurize(x).transpose(1, 2)
        xq, _, _, _ = self.rvq(xq)
        x = self._infer_from_codes(xq.transpose(1, 2))
        return x

    def encode(self, x: Tensor) -> Tensor:
        """Converts a Mel spectrogram into a sequence of tokens.

        Args:
            x: The input Mel spectrogram, with shape ``(B, T, C)``.

        Returns:
            The sequence of tokens, with shape ``(N, B, T)``.
        """
        xq = self._featurize(x)
        xq = self._pre_quant_to_tokens(xq)
        return xq

    def decode(self, tokens: Tensor) -> Tensor:
        """Converts a sequence of tokens to a Mel spectrogram.

        Args:
            tokens: The sequence of tokens, with shape ``(N, B, T)``.

        Returns:
            The decoded Mel spectrogram, with shape ``(B, T, C)``.
        """
        xq = self._tokens_to_embedding(tokens)
        x = self._infer_from_codes(xq)
        return x

    def quantizer(self) -> "MelCodecQuantizer":
        return MelCodecQuantizer(self)

    def dequantizer(self) -> "MelCodecDequantizer":
        return MelCodecDequantizer(self)


class MelCodecQuantizer(nn.Module):
    __constants__ = ["codebook_size", "num_mels", "encoder_causal"]

    def __init__(self, codec: MelCodec) -> None:
        super().__init__()

        self.codebook_size = codec.codebook_size
        self.num_mels = codec.num_mels
        self.encoder_causal = codec.encoder_causal

        # Copies the relevant attributes from the codec module.
        self.rvq = codec.rvq
        self.embs = codec.embs
        self.init_emb = codec.init_emb
        self.encoder_transformer = codec.encoder_transformer
        self.enc_in_proj = codec.enc_in_proj

        self.register_buffer("mask", codec.mask)

        self.mel_fn = AudioToHifiGanMels()

    mask: Tensor

    def _get_mels(self, audio: Tensor) -> Tensor:
        if audio.dim() == 3:
            assert audio.shape[1] == 1, "Expected mono audio."
            audio = audio.squeeze(1)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio_min, audio_max = audio.aminmax(dim=-1, keepdim=True)
        audio = audio / torch.maximum(audio_max, -audio_min).clamp_min(1e-2) * 0.999
        return self.mel_fn.wav_to_mels(audio.flatten(1)).transpose(1, 2)

    def _featurize(self, x: Tensor) -> Tensor:
        x = self.enc_in_proj(x)
        x = self.embs(x)
        if self.encoder_causal:
            x = self.encoder_transformer(x, mask=self.mask, is_causal=True)
        else:
            x = self.encoder_transformer(x, is_causal=False)
        return x

    def _pre_quant_to_tokens(self, xq: Tensor) -> Tensor:
        return self.rvq.encode(xq.transpose(1, 2))

    def encode(self, audio: Tensor) -> Tensor:
        """Converts a waveform to a set of tokens.

        Args:
            audio: The single-channel input waveform, with shape ``(B, T)``
                This should be at 22050 Hz.

        Returns:
            The quantized tokens, with shape ``(N, B, Tq)``
        """
        mels = self._get_mels(audio)
        xq = self._featurize(mels)
        xq = self._pre_quant_to_tokens(xq)
        return xq

    def forward(self, audio: Tensor) -> Tensor:
        return self.encode(audio)


class MelCodecDequantizer(nn.Module):
    def __init__(self, codec: MelCodec) -> None:
        super().__init__()

        self.codebook_size = codec.codebook_size
        self.num_mels = codec.num_mels
        self.encoder_causal = codec.encoder_causal

        # Copies the relevant attributes from the codec module.
        self.rvq = codec.rvq
        self.embs = codec.embs
        self.init_emb = codec.init_emb
        self.decoder_transformer = codec.decoder_transformer
        self.dec_in_proj = codec.dec_in_proj
        self.out_proj = codec.out_proj

        self.register_buffer("mask", codec.mask)

        self.hifigan = pretrained_hifigan()

    mask: Tensor

    def _get_audio(self, mels: Tensor) -> Tensor:
        return self.hifigan(mels.transpose(1, 2)).squeeze(1)

    def _tokens_to_embedding(self, tokens: Tensor) -> Tensor:
        return self.rvq.decode(tokens).transpose(1, 2)

    def _infer_from_codes(self, x_codes: Tensor) -> Tensor:
        init_emb = self.init_emb.repeat(x_codes.shape[0], 1, 1)
        x_nb = init_emb
        tsz = x_codes.shape[1]
        for t in range(tsz):
            x = x_codes[:, : x_nb.shape[1]] + x_nb
            x = self.decoder_transformer(x, mask=self.mask, is_causal=True)
            x = self.out_proj(x)
            if t < tsz - 1:
                x_nb = torch.cat([init_emb, self.dec_in_proj(x)], dim=1)
        return x

    def decode(self, tokens: Tensor) -> Tensor:
        """Converts a set of tokens to a waveform.

        Args:
            tokens: The single-channel input tokens, with shape ``(N, B, Tq)``,
                at 22050 Hz.

        Returns:
            The decoded waveform, with shape ``(B, T)``
        """
        xq = self._tokens_to_embedding(tokens)
        x = self._infer_from_codes(xq)
        return self._get_audio(x)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.decode(tokens)


def _load_pretrained_mel_codec(
    key: PretrainedMelCodecType,
    ckpt_url: str,
    sha256: str,
    load_weights: bool,
    config: MelCodecConfig,
) -> MelCodec:
    model = MelCodec(
        num_mels=config.num_mels,
        d_model=config.d_model,
        dim_feedforward=config.dim_feedforward,
        nhead=config.nhead,
        num_layers=config.num_layers,
        codebook_size=config.codebook_size,
        num_quantizers=config.num_quantizers,
        encoder_causal=config.encoder_causal,
    )

    if load_weights:
        model_fname = f"{key}.bin"

        with Timer("downloading checkpoint"):
            model_path = ensure_downloaded(ckpt_url, "codec", model_fname, sha256=sha256)

        with Timer("loading checkpoint", spinner=True):
            ckpt = safetensors.torch.load_file(model_path)
            model.load_state_dict(ckpt)

    return model


def pretrained_mel_codec(key: str | PretrainedMelCodecType, load_weights: bool = True, max_tsz: int = 2048) -> MelCodec:
    match key:
        case "librivox":
            return _load_pretrained_mel_codec(
                cast_pretrained_mel_codec_type(key),
                ckpt_url="https://huggingface.co/codekansas/codec/resolve/main/librivox.bin",
                sha256="7d884aebaf4ac1f56fb64191121a253a5f3446a1e4e68ad20ff94905158bdab3",
                load_weights=load_weights,
                config=MelCodecConfig(
                    num_mels=80,
                    d_model=768,
                    dim_feedforward=2048,
                    nhead=12,
                    num_layers=3,
                    codebook_size=1024,
                    num_quantizers=8,
                    encoder_causal=False,
                    max_tsz=max_tsz,
                ),
            )
        case _:
            raise ValueError(f"Unknown codec key: {key}")


def test_codec_adhoc() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Runs adhoc test of the codec.")
    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument("output_file", type=str, help="Path to output audio file")
    args = parser.parse_args()

    # Loads the audio file.
    audio, sr = torchaudio.load(args.input_file)
    audio = audio[:1]
    audio = audio[:, : sr * 10]
    if sr != 22050:
        audio = torchaudio.functional.resample(audio, sr, 22050)

    # Note: This normalizes the audio to the range [-1, 1], which may increase
    # the volume of the audio if it is quiet.
    audio = audio / audio.abs().max() * 0.999

    # Loads the pretrained model.
    model = pretrained_mel_codec("librivox")
    quantizer, dequantizer = model.quantizer(), model.dequantizer()
    tokens = quantizer(audio)
    audio = dequantizer(tokens)

    # Saves the audio.
    torchaudio.save(args.output_file, audio, 22050)

    logger.info("Saved %s", args.output_file)


if __name__ == "__main__":
    # python -m pretrained.mel_codec
    test_codec_adhoc()
