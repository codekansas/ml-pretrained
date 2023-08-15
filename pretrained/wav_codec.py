"""Defines a simple API for an audio quantizer model that runs on waveforms.

.. highlight:: python
.. code-block:: python

    from pretrained.wav_codec import pretrained_wav_codec

    model = pretrained_mel_codec("wav-codec-small")
    quantizer, dequantizer = model.quantizer(), model.dequantizer()

    # Convert some audio to a quantized representation.
    quantized = quantizer(audio)

    # Convert the quantized representation back to audio.
    audio = dequantizer(quantized)
"""

import argparse
import logging
from typing import Literal, cast, get_args

import torch
import torch.nn.functional as F
import torchaudio
import tqdm
from ml.models.codebook import ResidualVectorQuantization, VectorQuantization
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.device.auto import detect_device
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer
from torch import Tensor, nn, optim

logger = logging.getLogger(__name__)

PretrainedWavCodecType = Literal["wav-codec-small"]

RNNClass: type[nn.LSTM] | type[nn.GRU] = nn.LSTM
RNNState = tuple[Tensor, Tensor]


def cast_pretrained_mel_codec_type(s: str | PretrainedWavCodecType) -> PretrainedWavCodecType:
    if s not in get_args(PretrainedWavCodecType):
        raise KeyError(f"Invalid Codec type: {s} Expected one of: {get_args(PretrainedWavCodecType)}")
    return cast(PretrainedWavCodecType, s)


class Encoder(nn.Module):
    __constants__ = ["stride_length"]

    def __init__(self, stride_length: int, hidden_size: int) -> None:
        super().__init__()

        self.stride_length = stride_length
        self.in_proj = nn.Sequential(
            nn.Linear(stride_length, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

    def _split_waveform(self, waveform: Tensor, waveform_prev: Tensor | None) -> tuple[Tensor, Tensor]:
        if waveform_prev is not None:
            waveform = torch.cat((waveform_prev, waveform), dim=-1)
        tsz = waveform.shape[-1]
        rest = tsz % self.stride_length
        split = tsz - rest
        return waveform[..., :split], waveform[..., split:]

    def forward(
        self,
        waveform: Tensor,
        waveform_prev: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        waveform, waveform_rest = self._split_waveform(waveform, waveform_prev)
        x = waveform.unflatten(-1, (-1, self.stride_length))
        x = self.in_proj(x)
        return x, waveform_rest


class Decoder(nn.Module):
    def __init__(self, stride_length: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()

        self.stride_length = stride_length

        self.rnn = RNNClass(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, stride_length)

        self.waveform_proj = nn.Linear(stride_length, hidden_size)
        self.init_emb = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(
        self,
        toks: Tensor,
        waveform: Tensor,
        state: RNNState | None = None,
    ) -> tuple[Tensor, RNNState]:
        wemb = self.waveform_proj(waveform.unflatten(-1, (-1, self.stride_length)))
        wemb = torch.cat((self.init_emb.expand(wemb.shape[0], -1, -1), wemb[:, :-1]), dim=1)
        x = toks + wemb
        x, state_out = self.rnn(x, state)
        x = self.out_proj(x)
        x = x.flatten(-2)
        return x, state_out

    def infer(self, toks: Tensor, state: RNNState | None = None) -> tuple[Tensor, RNNState]:
        bsz, tsz, _ = toks.shape
        wemb = self.init_emb.expand(bsz, -1, -1)
        xs: list[Tensor] = []
        for t in range(tsz):
            x = toks[:, t : t + 1] + wemb
            x, state = self.rnn(x, state)
            x = self.out_proj(x)
            xs.append(x)
            if t < tsz - 1:
                wemb = self.waveform_proj(x)
        assert state is not None, "Empty tensor"
        x = torch.cat(xs, dim=1).flatten(1, 2)
        return x, state


class WavCodec(nn.Module):
    def __init__(
        self,
        stride_length: int,
        hidden_size: int,
        num_layers: int,
        codebook_size: int,
        num_quantizers: int,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(stride_length, hidden_size)
        self.decoder = Decoder(stride_length, hidden_size, num_layers)
        self.rvq = ResidualVectorQuantization(
            VectorQuantization(dim=hidden_size, codebook_size=codebook_size),
            num_quantizers=num_quantizers,
        )

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        x, _ = self.encoder(waveform)
        x, _, codebook_loss, _ = self.rvq(x.transpose(1, 2))
        x, _ = self.decoder(x.transpose(1, 2), waveform)
        return x, codebook_loss

    def encode(self, waveform: Tensor, waveform_prev: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x, waveform_rest = self.encoder(waveform, waveform_prev)
        x = self.rvq.encode(x.transpose(1, 2))
        return x, waveform_rest

    def decode(self, toks: Tensor, state: RNNState | None = None) -> tuple[Tensor, RNNState]:
        x: Tensor = self.rvq.decode(toks)
        x, state_out = self.decoder.infer(x.transpose(1, 2), state)
        return x, state_out

    def quantizer(self) -> "WavCodecQuantizer":
        return WavCodecQuantizer(self)

    def dequantizer(self) -> "WavCodecDequantizer":
        return WavCodecDequantizer(self)


class WavCodecQuantizer(nn.Module):
    def __init__(self, model: WavCodec) -> None:
        super().__init__()

        self.encoder = model.encoder
        self.rvq = model.rvq

    def encode(self, waveform: Tensor, waveform_prev: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Converts a waveform into a set of tokens.

        Args:
            waveform: The single-channel input waveform, with shape ``(B, T)``
                This should be at 16000 Hz.
            waveform_prev: The leftover from the previous call to ``encode``.

        Returns:
            The quantized tokens, with shape ``(N, B, Tq)``
        """
        x, waveform_rest = self.encoder(waveform, waveform_prev)
        x = self.rvq.encode(x.transpose(1, 2))
        return x, waveform_rest

    def forward(self, waveform: Tensor, waveform_prev: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self.encode(waveform, waveform_prev)


class WavCodecDequantizer(nn.Module):
    def __init__(self, model: WavCodec) -> None:
        super().__init__()

        self.decoder = model.decoder
        self.rvq = model.rvq

    def decode(self, toks: Tensor, state: RNNState | None = None) -> tuple[Tensor, RNNState]:
        """Converts a set of tokens into a waveform.

        Args:
            toks: The quantized tokens, with shape ``(N, B, Tq)``
            state: The previous state of the decoder.

        Returns:
            The single-channel output waveform, with shape ``(B, T)``, along
            with the new state of the decoder.
        """
        x: Tensor = self.rvq.decode(toks)
        x, state_out = self.decoder.infer(x.transpose(1, 2), state)
        return x, state_out

    def forward(self, toks: Tensor, state: RNNState | None = None) -> tuple[Tensor, RNNState]:
        return self.decode(toks, state)


def _load_pretrained_mel_codec(
    model: WavCodec,
    key: PretrainedWavCodecType,
    ckpt_url: str,
    sha256: str,
    load_weights: bool,
) -> WavCodec:
    if load_weights:
        model_fname = f"{key}.bin"

        with Timer("downloading checkpoint"):
            model_path = ensure_downloaded(ckpt_url, "codec", model_fname, sha256=sha256)

        with Timer("loading checkpoint", spinner=True):
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)

    return model


def pretrained_wav_codec(key: str | PretrainedWavCodecType, load_weights: bool = True) -> WavCodec:
    key = cast_pretrained_mel_codec_type(key)

    match key:
        case "wav-codec-small":
            return _load_pretrained_mel_codec(
                model=WavCodec(
                    stride_length=320,
                    hidden_size=512,
                    num_layers=2,
                    codebook_size=512,
                    num_quantizers=8,
                ),
                key=key,
                ckpt_url="https://huggingface.co/codekansas/codec/resolve/main/wav-codec-small.pt",
                sha256="2399d692805b76bf4f91c9a8ff984031f146b59c8750ac2ebc4d9cf0aa0fa835",
                load_weights=load_weights,
            )
        case _:
            raise ValueError(f"Unknown codec key: {key}")


def test_codec_adhoc() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Runs adhoc test of the codec.")
    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument("output_file", type=str, help="Path to output audio file")
    parser.add_argument("-k", "--key", choices=get_args(PretrainedWavCodecType), default="wav-codec-small")
    args = parser.parse_args()

    # Loads the audio file.
    audio, sr = torchaudio.load(args.input_file)
    audio = audio[:1]
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)

    # Note: This normalizes the audio to the range [-1, 1], which may increase
    # the volume of the audio if it is quiet.
    audio = audio / audio.abs().max() * 0.999

    # Loads the pretrained model.
    model = pretrained_wav_codec(args.key)
    quantizer, dequantizer = model.quantizer(), model.dequantizer()
    waveform_prev: Tensor | None = None
    decoder_state: RNNState | None = None
    audio_recs: list[Tensor] = []
    for audio_chunk in tqdm.tqdm(audio.split(16000 * 10, dim=-1)):
        tokens, waveform_prev = quantizer(audio_chunk, waveform_prev)
        audio_rec, decoder_state = dequantizer(tokens, decoder_state)
        audio_recs.append(audio_rec)

    # Saves the audio.
    audio = torch.cat(audio_recs, dim=-1)
    torchaudio.save(args.output_file, audio, 16000)

    logger.info("Saved %s", args.output_file)


def test_codec_training_adhoc() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Runs adhoc test of the codec.")
    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument("output_file", type=str, help="Path to output audio file")
    parser.add_argument("-n", "--num-steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument("-l", "--log-interval", type=int, default=1, help="Log interval")
    parser.add_argument("-s", "--num-seconds", type=float, default=5.0, help="Number of seconds to use")
    args = parser.parse_args()

    # Loads the audio file.
    audio, sr = torchaudio.load(args.input_file)
    audio = audio[:1]
    audio = audio[:, : int(sr * args.num_seconds)]
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)

    # Note: This normalizes the audio to the range [-1, 1], which may increase
    # the volume of the audio if it is quiet.
    audio = audio / audio.abs().max() * 0.999

    device = detect_device()
    audio = device.tensor_to(audio)

    # Loads the model.
    model = pretrained_wav_codec("wav-codec-small", load_weights=False)
    model.to(device._get_device())

    # Runs training.
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    with device.autocast_context():
        for i in range(args.num_steps):
            opt.zero_grad()
            rec_audio, codebook_loss = model(audio)
            loss = F.l1_loss(rec_audio, audio) + codebook_loss.sum()
            if torch.isnan(loss).any():
                logger.warning("NaN loss; aborting")
                break
            loss.backward()
            opt.step()

            if i % args.log_interval == 0:
                logger.info("Step %d: loss=%f", i, loss.item())

        rec_audio, _ = model(audio)
        rec_audio = rec_audio.detach().cpu().float()

    # Saves the audio.
    torchaudio.save(args.output_file, rec_audio, 16000)

    logger.info("Saved %s", args.output_file)


if __name__ == "__main__":
    # python -m pretrained.wav_codec
    test_codec_adhoc()
