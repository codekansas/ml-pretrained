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
from ml.models.codebook import ResidualVectorQuantization, VectorQuantization
from ml.models.modules import StreamingConv1d
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.device.auto import detect_device
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer, spinnerator
from torch import Tensor, nn, optim

logger = logging.getLogger(__name__)

PretrainedWavCodecType = Literal["base"]

RNNClass: type[nn.LSTM] | type[nn.GRU] = nn.LSTM
RNNState = tuple[Tensor, Tensor]

EncoderState = tuple[Tensor, list[tuple[Tensor, int]]]


def cast_pretrained_mel_codec_type(s: str | PretrainedWavCodecType) -> PretrainedWavCodecType:
    if s not in get_args(PretrainedWavCodecType):
        raise KeyError(f"Invalid Codec type: {s} Expected one of: {get_args(PretrainedWavCodecType)}")
    return cast(PretrainedWavCodecType, s)


def split_waveform(waveform: Tensor, stride_length: int, waveform_prev: Tensor | None = None) -> tuple[Tensor, Tensor]:
    if waveform_prev is not None:
        waveform = torch.cat((waveform_prev, waveform), dim=-1)
    tsz = waveform.shape[-1]
    rest = tsz % stride_length
    split = tsz - rest
    return waveform[..., :split], waveform[..., split:]


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
    __constants__ = ["stride_length"]

    def __init__(self, stride_length: int, d_model: int, kernel_size: int = 5) -> None:
        super().__init__()

        self.stride_length = stride_length

        self.encoder = nn.ModuleList(
            [
                CBR(stride_length, d_model, 1),
                CBR(d_model, d_model, kernel_size),
                CBR(d_model, d_model, kernel_size),
            ],
        )

    def forward(self, waveform: Tensor, state: EncoderState | None = None) -> tuple[Tensor, EncoderState]:
        waveform_prev = None if state is None else state[0]
        waveform, waveform_rest = split_waveform(waveform, self.stride_length, waveform_prev)
        x = waveform.unflatten(-1, (-1, self.stride_length)).transpose(1, 2)
        states_out: list[tuple[Tensor, int]] = []
        for i, conv in enumerate(self.encoder):
            x, state_out = conv(x, None if state is None else state[1][i])
            states_out.append(state_out)
        return x, (waveform_rest, states_out)


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

    def encode(self, waveform: Tensor, state: EncoderState | None = None) -> tuple[Tensor, EncoderState]:
        """Converts a waveform into a set of tokens.

        Args:
            waveform: The single-channel input waveform, with shape ``(B, T)``
                This should be at 16000 Hz.
            state: The encoder state from the previous step.

        Returns:
            The quantized tokens, with shape ``(N, B, Tq)``, along with the
            encoder state to pass to the next step.
        """
        x, state = self.encoder(waveform, state)
        x = self.rvq.encode(x)
        return x, state

    def forward(self, waveform: Tensor, state: EncoderState | None = None) -> tuple[Tensor, EncoderState]:
        return self.encode(waveform, state)


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
            model_path = ensure_downloaded(ckpt_url, "wav_codec", model_fname, sha256=sha256)

        with Timer("loading checkpoint", spinner=True):
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)

    return model


def pretrained_wav_codec(key: str | PretrainedWavCodecType, load_weights: bool = True) -> WavCodec:
    key = cast_pretrained_mel_codec_type(key)

    match key:
        case "base":
            return _load_pretrained_mel_codec(
                model=WavCodec(
                    stride_length=320,
                    hidden_size=1024,
                    num_layers=4,
                    codebook_size=1024,
                    num_quantizers=8,
                ),
                key=key,
                ckpt_url="https://huggingface.co/codekansas/codec/resolve/main/wavs_base.bin",
                sha256="cba9fce581adf9246926a492343e79e3783490d489e7e40ef56e0749d2f29834",
                load_weights=load_weights,
            )

        case _:
            raise ValueError(f"Unknown codec key: {key}")


def test_codec_adhoc() -> None:
    configure_logging()

    type_choices = list(get_args(PretrainedWavCodecType))

    parser = argparse.ArgumentParser(description="Runs adhoc test of the codec.")
    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument("output_file", type=str, help="Path to output audio file")
    parser.add_argument("-k", "--key", choices=type_choices, default=type_choices[0])
    parser.add_argument("-m", "--max-seconds", type=int)
    args = parser.parse_args()

    # Loads the audio file.
    audio, sr = torchaudio.load(args.input_file)
    audio = audio[:1]
    if args.max_seconds:
        audio = audio[:, : sr * args.max_seconds]
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)

    # Note: This normalizes the audio to the range [-1, 1], which may increase
    # the volume of the audio if it is quiet.
    audio = audio / audio.abs().max() * 0.999

    # Loads the pretrained model.
    model = pretrained_wav_codec(args.key)
    quantizer, dequantizer = model.quantizer(), model.dequantizer()
    encoder_state: EncoderState | None = None
    decoder_state: RNNState | None = None
    audio_recs: list[Tensor] = []
    for audio_chunk in spinnerator(audio.split(16000 * 10, dim=-1)):
        tokens, encoder_state = quantizer(audio_chunk, encoder_state)
        audio_rec, decoder_state = dequantizer(tokens, decoder_state)
        audio_recs.append(audio_rec)

    # Saves the audio.
    audio = torch.cat(audio_recs, dim=-1)
    torchaudio.save(args.output_file, audio, 16000)

    logger.info("Saved %s", args.output_file)


def test_codec_training_adhoc() -> None:
    configure_logging()

    type_choices = list(get_args(PretrainedWavCodecType))

    parser = argparse.ArgumentParser(description="Runs adhoc test of the codec.")
    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument("output_file", type=str, help="Path to output audio file")
    parser.add_argument("-t", "--type", choices=type_choices, default=type_choices[0])
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
    model = pretrained_wav_codec(args.type, load_weights=False)
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
