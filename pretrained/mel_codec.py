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
import functools
import logging
from typing import Literal, cast, get_args

import torch
import torchaudio
from ml.models.codebook import ResidualVectorQuantization, VectorQuantization
from ml.models.modules import StreamingConv1d
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.device.auto import detect_device
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer, spinnerator
from torch import Tensor, nn

from pretrained.vocoder.hifigan import HiFiGAN, PretrainedHiFiGANType, pretrained_hifigan

logger = logging.getLogger(__name__)

RNNClass: type[nn.LSTM] | type[nn.GRU] = nn.LSTM
RNNState = tuple[Tensor, Tensor]

PretrainedMelCodecType = Literal["base"]

EncoderState = list[tuple[Tensor, int]]


def cast_pretrained_mel_codec_type(s: str) -> PretrainedMelCodecType:
    if s not in get_args(PretrainedMelCodecType):
        raise KeyError(f"Invalid Codec type: {s} Expected one of: {get_args(PretrainedMelCodecType)}")
    return cast(PretrainedMelCodecType, s)


class CBR(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()

        self.conv = StreamingConv1d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor, state: tuple[Tensor, int] | None = None) -> tuple[Tensor, tuple[Tensor, int]]:
        x, state = self.conv.forward(x, state)
        x = self.bn(x)
        x = self.act(x)
        return x, state


class Encoder(nn.Module):
    """Defines the encoder module.

    This module takes the Mel spectrogram as an input and outputs the
    latent representation to quantize.

    Parameters:
        num_mels: Number of input Mel spectrogram bins.
        d_model: The hidden dimension of the model.

    Inputs:
        mels: The input Mel spectrogram, with shape ``(B, T, C)``.
        state: The previous state of the encoder, if any.

    Outputs:
        The latent representation, with shape ``(B, T, C)``, along with the
        updated state.
    """

    def __init__(self, num_mels: int, d_model: int) -> None:
        super().__init__()

        self.encoder = nn.ModuleList(
            [
                CBR(num_mels, d_model, 1),
                CBR(d_model, d_model, 5),
                CBR(d_model, d_model, 5),
            ],
        )

    def forward(self, mels: Tensor, state: EncoderState | None = None) -> tuple[Tensor, EncoderState]:
        x = mels.transpose(1, 2)
        states_out: EncoderState = []
        for i, layer in enumerate(self.encoder):
            x, state_out = layer(x, None if state is None else state[i])
            states_out.append(state_out)
        return x.transpose(1, 2), states_out


class Decoder(nn.Module):
    """Defines the decoder module.

    This module takes the latent representation as input and outputs the
    reconstructed Mel spectrogram. this can be run in inference mode, where
    the model expects to see batches of codes and maintains a state over time,
    or in training mode, where the model expects to see the ground truth
    Mel spectrogram in addition to the codes.

    Parameters:
        num_mels: Number of input Mel spectrogram bins.
        d_model: The hidden dimension of the model.

    Inputs:
        codes: The latent representation, with shape ``(B, T, C)``.
        mels: The input Mel spectrogram, with shape ``(B, T, C)``, if in
            training mode.

    Outputs:
        The reconstructed Mel spectrogram, with shape ``(B, T, C)``, along
        with the updated state if in training mode.
    """

    def __init__(self, num_mels: int, d_model: int, num_layers: int) -> None:
        super().__init__()

        self.register_buffer("init_emb", torch.zeros(1, 1, d_model))

        self.decoder_rnn = RNNClass(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
        )

        self.in_proj = nn.Linear(num_mels, d_model)
        self.out_proj = nn.Linear(d_model, num_mels)

    init_emb: Tensor

    def forward(self, codes: Tensor, mels: Tensor) -> Tensor:
        x_prev = self.in_proj(mels[:, :-1])
        x_prev = torch.cat([self.init_emb.repeat(x_prev.shape[0], 1, 1), x_prev], dim=1)
        x = codes + x_prev
        x, _ = self.decoder_rnn(x)
        x = self.out_proj(x)
        return x

    def infer(self, codes: Tensor, state: RNNState | None = None) -> tuple[Tensor, RNNState]:
        init_emb = self.init_emb.repeat(codes.shape[0], 1, 1)
        x_nb = init_emb
        tsz = codes.shape[1]
        xs = []
        for t in range(tsz):
            x = codes[:, t : t + 1] + x_nb
            x, state = self.decoder_rnn(x, state)
            x = self.out_proj(x)
            xs.append(x)
            if t < tsz - 1:
                x_nb = self.in_proj(x)
        assert state is not None
        return torch.cat(xs, dim=1), state


class MelCodec(nn.Module):
    """Defines an audio RNN module.

    This module takes the Mel spectrogram as an input and outputs the predicted
    next step of the Mel spectrogram.

    Parameters:
        num_mels: Number of input Mel spectrogram bins.
        d_model: The hidden dimension of the model.
        num_layers: Number of hidden layers in the decoder.
        codebook_size: Number of codebook entries.
        num_quantizers: Number of quantizers to use.
    """

    __constants__ = ["codebook_size", "num_mels", "hifigan_key"]

    def __init__(
        self,
        num_mels: int,
        d_model: int,
        num_layers: int,
        codebook_size: int,
        num_quantizers: int,
        hifigan_key: PretrainedHiFiGANType,
    ) -> None:
        super().__init__()

        self.codebook_size = codebook_size
        self.num_mels = num_mels
        self.hifigan_key = hifigan_key

        self.rvq = ResidualVectorQuantization(
            VectorQuantization(
                dim=d_model,
                codebook_size=codebook_size,
                kmeans_init=False,
            ),
            num_quantizers=num_quantizers,
        )

        self.encoder = Encoder(num_mels, d_model)
        self.decoder = Decoder(num_mels, d_model, num_layers)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Runs the forward pass of the model.

        Args:
            x: The input Mel spectrogram, with shape ``(B, T, C)``.

        Returns:
            The predicted next step of the Mel spectrogram, with shape
            ``(B, T, C)``, along with the codebook loss.
        """
        xq = self.encoder(x).transpose(1, 2)
        xq, _, codebook_loss, _ = self.rvq(xq)
        x = self.decoder(xq.transpose(1, 2), x)
        return x, codebook_loss

    def infer(self, x: Tensor) -> Tensor:
        """Runs the inference pass, for evaluating model quality.

        This just converts the input mels to codes and then decodes them.

        Args:
            x: The input Mel spectrogram, with shape ``(B, T, C)``.

        Returns:
            The predicted next step of the Mel spectrogram, with shape
            ``(B, T, C)``,
        """
        xq = self.encoder(x).transpose(1, 2)
        xq, _, _, _ = self.rvq(xq)
        x, _ = self.decoder.infer(xq.transpose(1, 2))
        return x

    @functools.cached_property
    def hifigan(self) -> "HiFiGAN":
        return pretrained_hifigan(self.hifigan_key)

    def quantizer(self) -> "MelCodecQuantizer":
        return MelCodecQuantizer(self, self.hifigan)

    def dequantizer(self) -> "MelCodecDequantizer":
        return MelCodecDequantizer(self, self.hifigan)


class MelCodecQuantizer(nn.Module):
    __constants__ = ["codebook_size", "num_mels"]

    def __init__(self, codec: MelCodec, hifigan: HiFiGAN) -> None:
        super().__init__()

        self.codebook_size = codec.codebook_size
        self.num_mels = codec.num_mels

        # Copies the relevant attributes from the codec module.
        self.rvq = codec.rvq
        self.encoder = codec.encoder

        self.mel_fn = hifigan.audio_to_mels()

    def _get_mels(self, audio: Tensor) -> Tensor:
        if audio.dim() == 3:
            assert audio.shape[1] == 1, "Expected mono audio."
            audio = audio.squeeze(1)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio_min, audio_max = audio.aminmax(dim=-1, keepdim=True)
        audio = audio / torch.maximum(audio_max, -audio_min).clamp_min(1e-2) * 0.999
        return self.mel_fn.wav_to_mels(audio.flatten(1)).transpose(1, 2)

    def _featurize(self, mels: Tensor, state: EncoderState | None = None) -> tuple[Tensor, EncoderState]:
        return self.encoder(mels, state)

    def _pre_quant_to_tokens(self, xq: Tensor) -> Tensor:
        return self.rvq.encode(xq.transpose(1, 2))

    def encode(self, audio: Tensor, state: EncoderState | None = None) -> tuple[Tensor, EncoderState]:
        """Converts a waveform to a set of tokens.

        Args:
            audio: The single-channel input waveform, with shape ``(B, T)``
                This should be at 22050 Hz.
            state: The encoder state from the previous step, if any.

        Returns:
            The quantized tokens, with shape ``(N, B, Tq)``, along with the
            updated encoder state.
        """
        mels = self._get_mels(audio)
        xq, state = self._featurize(mels, state)
        xq = self._pre_quant_to_tokens(xq)
        return xq, state

    def forward(self, audio: Tensor, state: EncoderState | None = None) -> tuple[Tensor, EncoderState]:
        return self.encode(audio, state)


class MelCodecDequantizer(nn.Module):
    def __init__(self, codec: MelCodec, hifigan: HiFiGAN) -> None:
        super().__init__()

        self.codebook_size = codec.codebook_size
        self.num_mels = codec.num_mels

        # Copies the relevant attributes from the codec module.
        self.rvq = codec.rvq
        self.decoder = codec.decoder

        # Stores the HiFiGAN model for converting from Mels back to audio.
        self.hifigan = hifigan

    def _get_audio(self, mels: Tensor) -> Tensor:
        return self.hifigan.infer(mels.transpose(1, 2)).squeeze(1)

    def _tokens_to_embedding(self, tokens: Tensor) -> Tensor:
        return self.rvq.decode(tokens).transpose(1, 2)

    def _infer_from_codes(self, codes: Tensor, state: RNNState | None = None) -> tuple[Tensor, RNNState]:
        return self.decoder.infer(codes, state)

    def decode(self, tokens: Tensor, state: RNNState | None = None) -> tuple[Tensor, RNNState]:
        """Converts a set of tokens to a waveform.

        Args:
            tokens: The single-channel input tokens, with shape ``(N, B, Tq)``,
                at 22050 Hz.
            state: The decoder state from the previous step, if any.

        Returns:
            The decoded waveform, with shape ``(B, T)``, along with the updated
            decoder state.
        """
        xq = self._tokens_to_embedding(tokens)
        x, state = self._infer_from_codes(xq, state)
        return self._get_audio(x), state

    def forward(self, tokens: Tensor, state: RNNState | None = None) -> tuple[Tensor, RNNState]:
        return self.decode(tokens)


def _load_pretrained_mel_codec(
    model: MelCodec,
    key: PretrainedMelCodecType,
    ckpt_url: str,
    sha256: str,
    load_weights: bool,
) -> MelCodec:
    if load_weights:
        model_fname = f"{key}.bin"

        with Timer("downloading checkpoint"):
            model_path = ensure_downloaded(ckpt_url, "mel_codec", model_fname, sha256=sha256)

        with Timer("loading checkpoint", spinner=True):
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)

    return model


def pretrained_mel_codec(key: str | PretrainedMelCodecType, load_weights: bool = True) -> MelCodec:
    key = cast_pretrained_mel_codec_type(key)

    match key:
        case "base":
            return _load_pretrained_mel_codec(
                model=MelCodec(
                    num_mels=128,
                    d_model=768,
                    num_layers=3,
                    codebook_size=1024,
                    num_quantizers=8,
                    hifigan_key="16000hz",
                ),
                key=key,
                ckpt_url="https://huggingface.co/codekansas/codec/resolve/main/mels_base.bin",
                sha256="1a693724fa59be9e6d113a5bbd10a6281583ed4c4779f9559dbc2f74166e8c28",
                load_weights=load_weights,
            )

        case _:
            raise ValueError(f"Unknown codec key: {key}")


def test_codec_adhoc() -> None:
    configure_logging()

    type_choices = list(get_args(PretrainedMelCodecType))

    parser = argparse.ArgumentParser(description="Runs adhoc test of the codec.")
    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument("output_file", type=str, help="Path to output audio file")
    parser.add_argument("-k", "--key", choices=type_choices, default=type_choices[0])
    parser.add_argument("-m", "--max-seconds", type=int)
    args = parser.parse_args()

    dev = detect_device()

    # Loads the pretrained model.
    model = pretrained_mel_codec(args.key)
    quantizer, dequantizer = model.quantizer(), model.dequantizer()
    dev.module_to(quantizer)
    dev.module_to(dequantizer)

    # Loads the audio file.
    audio, sr = torchaudio.load(args.input_file)
    audio = audio[:1]
    if args.max_seconds:
        audio = audio[:, : sr * args.max_seconds]
    tsr = dequantizer.hifigan.sampling_rate
    if sr != tsr:
        audio = torchaudio.functional.resample(audio, sr, tsr)

    # Note: This normalizes the audio to the range [-1, 1], which may increase
    # the volume of the audio if it is quiet.
    audio = audio / audio.abs().max() * 0.999
    audio = dev.tensor_to(audio)

    # Encodes the audio sequence to tokens.
    encoder_state: EncoderState | None = None
    for audio_chunk in spinnerator(audio.split(tsr * 3, dim=1), desc="Encoding"):
        tokens, encoder_state = quantizer(audio, encoder_state)

    # Decodes the tokens to audio.
    audio_chunks: list[Tensor] = []
    decoder_state: RNNState | None = None
    for token_chunk in spinnerator(tokens.split(50, dim=-1), desc="Decoding"):
        audio_chunk, decoder_state = dequantizer(token_chunk, decoder_state)
        audio_chunks.append(audio_chunk)

    # Saves the audio.
    torchaudio.save(args.output_file, audio.cpu(), dequantizer.hifigan.sampling_rate)

    logger.info("Saved %s", args.output_file)


if __name__ == "__main__":
    # python -m pretrained.mel_codec
    test_codec_adhoc()
