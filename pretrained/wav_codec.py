"""Defines a simple API for an audio quantizer model that runs on waveforms.

.. highlight:: python
.. code-block:: python

    from pretrained.wav_codec import pretrained_wav_codec

    model = pretrained_wav_codec("librivox")
    encoder, decoder = model.encoder(), model.decoder()

    # Convert some audio to a quantized representation.
    audio_leftover = None
    all_tokens = []
    for audio_chunk in audio_chunks:
        if audio_leftover is not None:
            audio_chunk = torch.cat([audio_leftover, audio_chunk], dim=1)
        tokens, audio_leftover = encoder.encode(audio_chunk)
        all_tokens.append(tokens)

    # Convert the quantized representation back to audio.
    rnn_states = None
    audio_chunks = []
    for tokens in all_tokens:
        audio_chunk, rnn_states = decoder.decode(tokens, rnn_states)
        audio_chunks.append(audio_chunk)
    reconstructed_audio = torch.cat(audio_chunks, dim=1)
"""

import argparse
import logging
import math
import warnings
from typing import Literal, NamedTuple, cast, get_args

import safetensors.torch
import torch
import torchaudio
from ml.models.activations import ActivationType, get_activation
from ml.models.codebook import ResidualVectorQuantization, VectorQuantization
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer
from torch import Tensor, nn

logger = logging.getLogger(__name__)

DEFAULT_CONV_DIM: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
DEFAULT_CONV_STRIDE: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)

PretrainedWavCodecSize = Literal["librivox"]


def cast_pretrained_mel_codec_type(s: str) -> PretrainedWavCodecSize:
    if s not in get_args(PretrainedWavCodecSize):
        raise KeyError(f"Invalid Codec type: {s} Expected one of: {get_args(PretrainedWavCodecSize)}")
    return cast(PretrainedWavCodecSize, s)


class WavCodecState(NamedTuple):
    waveform_leftover: Tensor
    rnn_states: tuple[Tensor, Tensor]


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel: int,
        bias: bool = True,
        feat_extract_activation: ActivationType = "gelu",
    ) -> None:
        super().__init__()

        self.in_conv_dim = in_channels
        self.out_conv_dim = out_channels

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=kernel,
            stride=stride,
            bias=bias,
        )
        self.activation = get_activation(feat_extract_activation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class ConvTransposeLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel: int,
        bias: bool = True,
        feat_extract_activation: ActivationType = "gelu",
    ) -> None:
        super().__init__()

        self.in_conv_dim = in_channels
        self.out_conv_dim = out_channels

        self.conv = nn.ConvTranspose1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=kernel,
            stride=stride,
            bias=bias,
        )
        self.activation = get_activation(feat_extract_activation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        conv_dim: tuple[int, ...] = DEFAULT_CONV_DIM,
        conv_stride: tuple[int, ...] = DEFAULT_CONV_STRIDE,
        conv_bias: bool = True,
        feat_extract_activation: ActivationType = "gelu",
    ) -> None:
        super().__init__()

        assert len(conv_dim) == len(conv_stride)
        num_feat_extract_layers = len(conv_dim)

        conv_layers: list[nn.Module] = []
        for i in range(num_feat_extract_layers):
            conv_layers += [
                ConvLayer(
                    in_channels=1 if i == 0 else conv_dim[i - 1],
                    out_channels=conv_dim[i],
                    stride=conv_stride[i],
                    kernel=conv_stride[i],
                    bias=conv_bias,
                    feat_extract_activation=feat_extract_activation,
                ),
            ]
        conv_layers += [
            ConvLayer(
                in_channels=conv_dim[-1],
                out_channels=hidden_size,
                stride=1,
                kernel=1,
                bias=conv_bias,
                feat_extract_activation=feat_extract_activation,
            ),
        ]
        self.conv_layers = nn.ModuleList(conv_layers)

    def _freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_values: Tensor) -> Tensor:
        hidden_states = input_values[:, None]
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        conv_dim: tuple[int, ...] = DEFAULT_CONV_DIM,
        conv_stride: tuple[int, ...] = DEFAULT_CONV_STRIDE,
        conv_bias: bool = True,
        feat_extract_activation: ActivationType = "gelu",
    ) -> None:
        super().__init__()

        assert len(conv_dim) == len(conv_stride)
        num_feat_extract_layers = len(conv_dim)

        conv_layers: list[nn.Module] = []
        conv_layers += [
            ConvLayer(
                in_channels=hidden_size,
                out_channels=conv_dim[0],
                stride=1,
                kernel=1,
                bias=conv_bias,
                feat_extract_activation=feat_extract_activation,
            ),
        ]
        for i in range(num_feat_extract_layers - 1, -1, -1):
            conv_layers += [
                ConvTransposeLayer(
                    in_channels=conv_dim[i],
                    out_channels=1 if i == 0 else conv_dim[i - 1],
                    stride=conv_stride[i],
                    kernel=conv_stride[i],
                    bias=conv_bias,
                    feat_extract_activation=feat_extract_activation,
                ),
            ]
        self.conv_layers = nn.ModuleList(conv_layers)

    def _freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, hidden_states: Tensor) -> Tensor:
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        output_values = hidden_states.squeeze(1)
        return output_values


def clip_waveform(
    waveform: Tensor,
    receptive_field_size: int,
    waveform_leftover: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if waveform_leftover is not None:
        waveform = torch.cat([waveform_leftover, waveform], dim=1)
    _, tsz = waveform.shape
    tsz_leftover = tsz % receptive_field_size
    waveform, waveform_leftover_out = waveform[:, : tsz - tsz_leftover], waveform[:, tsz - tsz_leftover :]
    return waveform, waveform_leftover_out


class WavCodec(nn.Module):
    __constants__ = ["receptive_field_size"]

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        codebook_size: int,
        num_quantizers: int,
        conv_dim: tuple[int, ...] = DEFAULT_CONV_DIM,
        conv_stride: tuple[int, ...] = DEFAULT_CONV_STRIDE,
        conv_bias: bool = True,
        feat_extract_activation: ActivationType = "gelu",
    ) -> None:
        super().__init__()

        self.receptive_field_size = math.prod(conv_stride)

        self.extractor = Encoder(
            hidden_size=hidden_size,
            conv_dim=conv_dim,
            conv_stride=conv_stride,
            conv_kernel=conv_stride,
            conv_bias=conv_bias,
            feat_extract_activation=feat_extract_activation,
        )

        self.autoregressive_extractor = Encoder(
            hidden_size=hidden_size,
            conv_dim=conv_dim,
            conv_stride=conv_stride,
            conv_kernel=conv_stride,
            conv_bias=conv_bias,
            feat_extract_activation=feat_extract_activation,
        )

        self.quantizer = ResidualVectorQuantization(
            VectorQuantization(dim=hidden_size, codebook_size=codebook_size),
            num_quantizers=num_quantizers,
        )

        self.init_state = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.wav_decoder = Decoder(
            hidden_size=hidden_size,
            conv_dim=conv_dim,
            conv_stride=conv_stride,
            conv_kernel=conv_stride,
            conv_bias=conv_bias,
            feat_extract_activation=feat_extract_activation,
        )

    def forward(self, waveform: Tensor, state: WavCodecState | None = None) -> tuple[Tensor, Tensor, WavCodecState]:
        waveform_leftover_prev = None if state is None else state.waveform_leftover
        waveform, waveform_leftover = clip_waveform(waveform, self.receptive_field_size, waveform_leftover_prev)

        # Extracts waveform features.
        x: Tensor = self.extractor(waveform)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        # Runs the quantizer.
        xq: Tensor = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        xq, _, vq_loss, _ = self.quantizer(xq)
        xq = xq.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        # Extracts autoregressive waveform features.
        xa: Tensor = self.autoregressive_extractor(waveform)
        xa = xa.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        # Shifts the encoded waveform by one timestep and adds quantized values.
        bsz = xa.shape[0]
        xa = torch.cat([self.init_state.expand(bsz, -1, -1), xa[:, :-1, :]], dim=1)
        x = xa + xq

        # Runs the RNN.
        rnn_states = None if state is None else state.rnn_states
        x, rnn_states_out = self.rnn(x, rnn_states)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        # Decodes to a waveform.
        x = self.wav_decoder(x)

        # Gets the new state.
        state_out = WavCodecState(
            waveform_leftover=waveform_leftover,
            rnn_states=rnn_states_out,
        )

        return x, vq_loss, state_out

    def encoder(self) -> "WavCodecEncoder":
        return WavCodecEncoder(self)

    def decoder(self) -> "WavCodecDecoder":
        return WavCodecDecoder(self)


class WavCodecEncoder(nn.Module):
    __constants__ = ["receptive_field_size"]

    def __init__(self, codec: WavCodec) -> None:
        super().__init__()

        self.receptive_field_size = codec.receptive_field_size

        self.extractor = codec.extractor
        self.quantizer = codec.quantizer

    def encode(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        """Encodes a given waveform into tensors.

        Args:
            waveform: The input waveform, with shape (B, T)

        Returns:
            The encoded tokens, with shape (num_quantizers, B, T), and the
            leftover part of the waveform, with shape (B, T_leftover).
        """
        waveform, waveform_leftover = clip_waveform(waveform, self.receptive_field_size)

        # Extracts waveform features.
        x: Tensor = self.extractor(waveform)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        # Quantizes to get tokens.
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = self.quantizer.encode(x)

        return x, waveform_leftover

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        return self.encode(waveform)


class WavCodecDecoder(nn.Module):
    def __init__(self, codec: WavCodec) -> None:
        super().__init__()

        self.autoregressive_extractor = codec.autoregressive_extractor
        self.quantizer = codec.quantizer
        self.init_state = codec.init_state
        self.rnn = codec.rnn
        self.wav_decoder = codec.wav_decoder

    def decode(
        self,
        tokens: Tensor,
        rnn_states: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Decodes a given waveform from tokens.

        Args:
            tokens: The input tokens, with shape (num_quantizers, B, T).
            rnn_states: The RNN states, with shape (num_layers, B, hidden_size).

        Returns:
            The decoded waveform, with shape (B, T), and the next RNN states,
            with shape (num_layers, B, hidden_size).
        """
        xq: Tensor = self.quantizer.decode(tokens)
        xq = xq.transpose(1, 2)

        bsz, tsz, _ = xq.shape
        x: Tensor = self.init_state.expand(bsz, -1, -1)
        xs: list[Tensor] = []
        for t in range(tsz):
            x = x + xq[:, t : t + 1, :]

            # Runs the RNN.
            x, rnn_states = self.rnn(x, rnn_states)
            x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

            # Decodes to a waveform.
            x = self.wav_decoder(x)
            xs.append(x)

            if t < tsz - 1:
                # Extracts waveform features.
                x = self.autoregressive_extractor(x)
                x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        x = torch.cat(xs, dim=1)
        assert rnn_states is not None, "Empty tensor"
        return x, rnn_states

    def forward(
        self,
        tokens: Tensor,
        rnn_states: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        return self.decode(tokens, rnn_states)


def _load_pretrained_wav_codec(
    size: PretrainedWavCodecSize,
    ckpt_url: str,
    sha256: str,
    num_layers: int,
    hidden_size: int,
    codebook_size: int,
    num_quantizers: int,
    load_weights: bool = True,
    conv_dim: tuple[int, ...] = DEFAULT_CONV_DIM,
    conv_stride: tuple[int, ...] = DEFAULT_CONV_STRIDE,
    conv_bias: bool = True,
    feat_extract_activation: ActivationType = "gelu",
) -> WavCodec:
    with Timer("building empty model", spinner=True):
        model = WavCodec(
            num_layers=num_layers,
            hidden_size=hidden_size,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            conv_dim=conv_dim,
            conv_stride=conv_stride,
            conv_bias=conv_bias,
            feat_extract_activation=feat_extract_activation,
        )

    # Loads the model weights.
    if load_weights:
        model_fname = f"{size}.bin"

        with Timer("downloading checkpoint"):
            model_path = ensure_downloaded(ckpt_url, "wav-codec", model_fname, sha256=sha256)

        with Timer("loading checkpoint", spinner=True):
            ckpt = safetensors.torch.load_file(model_path, device="cpu")
            model.load_state_dict(ckpt)

    return model


def pretrained_wav_codec(size: PretrainedWavCodecSize, load_weights: bool = True) -> WavCodec:
    match size:
        case "librivox":
            if load_weights:
                warnings.warn("No pretrained weights available for `librivox` get, using random weights")

            return _load_pretrained_wav_codec(
                size=size,
                ckpt_url="",
                sha256="",
                num_layers=4,
                hidden_size=512,
                codebook_size=1024,
                num_quantizers=8,
                load_weights=False,
            )

        case _:
            raise NotImplementedError(f"Invalid size: {size}")


def test_wav_codec() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Runs adhoc test of the codec.")
    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument("output_file", type=str, help="Path to output audio file")
    parser.add_argument("-k", "--key", choices=get_args(PretrainedWavCodecSize), default="librivox")
    args = parser.parse_args()

    # Loads the audio file.
    audio, sr = torchaudio.load(args.input_file)
    audio = audio[:1]
    audio = audio[:, : sr * 10]
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)

    # Note: This normalizes the audio to the range [-1, 1], which may increase
    # the volume of the audio if it is quiet.
    audio = audio / audio.abs().max() * 0.999

    # Runs the codec.
    model = pretrained_wav_codec(args.key)
    encoder, decoder = model.encoder(), model.decoder()
    tokens, _ = encoder.encode(audio)
    audio, _ = decoder.decode(tokens)

    # Saves the audio.
    torchaudio.save(args.output_file, audio, 16000)

    logger.info("Saved %s", args.output_file)


if __name__ == "__main__":
    # python -m pretrained.wav_codec
    test_wav_codec()
