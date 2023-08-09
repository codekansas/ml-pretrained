"""Defines a simple API for an audio quantizer model that runs on waveforms.

.. highlight:: python
.. code-block:: python

    from pretrained.wav_codec import pretrained_wav_codec

    model = pretrained_wav_codec("librivox")
    quantizer, dequantizer = model.quantizer(), model.dequantizer()

    # Convert some audio to a quantized representation.
    quantized = quantizer(audio)

    # Convert the quantized representation back to audio.
    audio = dequantizer(quantized)
"""

import argparse
import logging
import math
import warnings
from typing import Literal, NamedTuple, cast, get_args

import safetensors.torch
import torch
from ml.models.activations import ActivationType, get_activation
from ml.models.codebook import ResidualVectorQuantization, VectorQuantization
from ml.models.norms import ConvLayerNorm
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer
from torch import Tensor, nn

from pretrained.hubert import HubertFeatureEncoder, HubertFeatureProjection

logger = logging.getLogger(__name__)

DEFAULT_CONV_DIM: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
DEFAULT_CONV_STRIDE: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
DEFAULT_CONV_KERNEL: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)

PretrainedWavCodecSize = Literal["librivox"]


def cast_pretrained_mel_codec_type(s: str) -> PretrainedWavCodecSize:
    if s not in get_args(PretrainedWavCodecSize):
        raise KeyError(f"Invalid Codec type: {s} Expected one of: {get_args(PretrainedWavCodecSize)}")
    return cast(PretrainedWavCodecSize, s)


class WavCodecState(NamedTuple):
    waveform_leftover: Tensor
    rnn_states: tuple[Tensor, Tensor]


class GroupNormConvTransposeLayer(nn.Module):
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
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)
        self.activation = get_activation(feat_extract_activation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class NoLayerNormConvTransposeLayer(nn.Module):
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


class LayerNormConvTransposeLayer(nn.Module):
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
        self.layer_norm = ConvLayerNorm(self.out_conv_dim, dims=1)
        self.activation = get_activation(feat_extract_activation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Decoder(nn.Module):
    def __init__(
        self,
        conv_dim: tuple[int, ...] = DEFAULT_CONV_DIM,
        conv_stride: tuple[int, ...] = DEFAULT_CONV_STRIDE,
        conv_kernel: tuple[int, ...] = DEFAULT_CONV_KERNEL,
        conv_bias: bool = True,
        feat_extract_norm: Literal["group", "layer"] = "layer",
        feat_extract_activation: ActivationType = "gelu",
    ) -> None:
        super().__init__()

        assert len(conv_dim) == len(conv_stride)
        num_feat_extract_layers = len(conv_dim)

        conv_layers: list[nn.Module] = []
        if feat_extract_norm == "group":
            conv_layers += [
                GroupNormConvTransposeLayer(
                    in_channels=conv_dim[-1],
                    out_channels=conv_dim[-2],
                    stride=conv_stride[-1],
                    kernel=conv_kernel[-1],
                    bias=conv_bias,
                    feat_extract_activation=feat_extract_activation,
                )
            ]
            for i in range(num_feat_extract_layers - 2, -1, -1):
                conv_layers += [
                    NoLayerNormConvTransposeLayer(
                        in_channels=conv_dim[i],
                        out_channels=1 if i == 0 else conv_dim[i - 1],
                        stride=conv_stride[i],
                        kernel=conv_kernel[i],
                        bias=conv_bias,
                        feat_extract_activation=feat_extract_activation,
                    )
                ]
        elif feat_extract_norm == "layer":
            for i in range(num_feat_extract_layers - 1, -1, -1):
                conv_layers += [
                    LayerNormConvTransposeLayer(
                        in_channels=conv_dim[i],
                        out_channels=1 if i == 0 else conv_dim[i - 1],
                        stride=conv_stride[i],
                        kernel=conv_kernel[i],
                        bias=conv_bias,
                        feat_extract_activation=feat_extract_activation,
                    )
                ]
        else:
            raise ValueError(f"{feat_extract_norm=}, but has to be one of ['group', 'layer']")
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
        feat_extract_norm: Literal["group", "layer"] = "layer",
        feat_extract_activation: ActivationType = "gelu",
        layer_norm_eps: float = 1e-5,
        feat_proj_dropout: float = 0.0,
        feat_proj_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.receptive_field_size = math.prod(conv_stride)

        self.extractor = HubertFeatureEncoder(
            conv_dim=conv_dim,
            conv_stride=conv_stride,
            conv_kernel=conv_stride,
            conv_bias=conv_bias,
            feat_extract_norm=feat_extract_norm,
            feat_extract_activation=feat_extract_activation,
        )

        self.input_projector = HubertFeatureProjection(
            input_size=conv_dim[-1],
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            feat_proj_dropout=feat_proj_dropout,
            feat_proj_layer_norm=feat_proj_layer_norm,
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

        self.output_projector = HubertFeatureProjection(
            input_size=hidden_size,
            hidden_size=conv_dim[-1],
            layer_norm_eps=layer_norm_eps,
            feat_proj_dropout=feat_proj_dropout,
            feat_proj_layer_norm=feat_proj_layer_norm,
        )

        self.wav_decoder = Decoder(
            conv_dim=conv_dim,
            conv_stride=conv_stride,
            conv_kernel=conv_stride,
            conv_bias=conv_bias,
            feat_extract_norm=feat_extract_norm,
            feat_extract_activation=feat_extract_activation,
        )

    def forward(self, waveform: Tensor, state: WavCodecState | None = None) -> tuple[Tensor, Tensor, WavCodecState]:
        waveform_leftover_prev = None if state is None else state.waveform_leftover
        waveform, waveform_leftover = clip_waveform(waveform, self.receptive_field_size, waveform_leftover_prev)

        # Extracts waveform features.
        x: Tensor = self.extractor(waveform)

        # Projects to the RNN space.
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.input_projector(x)

        # Runs the quantizer.
        xq: Tensor = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        xq, _, vq_loss, _ = self.quantizer(xq)
        xq = xq.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        # Shifts the encoded waveform by one timestep and adds quantized values.
        bsz = x.shape[0]
        x = torch.cat([self.init_state.expand(bsz, -1, -1), x[:, :-1, :]], dim=1)
        x = x + xq

        # Runs the RNN.
        rnn_states = None if state is None else state.rnn_states
        x, rnn_states_out = self.rnn(x, rnn_states)

        # Projects to the output space.
        x = self.output_projector(x)
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
        self.input_projector = codec.input_projector
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

        # Projects to the RNN space.
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.input_projector(x)

        # Quantizes to get tokens.
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = self.quantizer.encode(x)

        return x, waveform_leftover

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        return self.encode(waveform)


class WavCodecDecoder(nn.Module):
    def __init__(self, codec: WavCodec) -> None:
        super().__init__()

        self.extractor = codec.extractor
        self.input_projector = codec.input_projector
        self.quantizer = codec.quantizer
        self.init_state = codec.init_state
        self.rnn = codec.rnn
        self.output_projector = codec.output_projector
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

            # Projects to the output space.
            x = self.output_projector(x)
            x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

            # Decodes to a waveform.
            x = self.wav_decoder(x)
            xs.append(x)

            if t < tsz - 1:
                # Extracts waveform features.
                x = self.extractor(x)

                # Projects to the RNN space.
                x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
                x = self.input_projector(x)

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
    feat_extract_norm: Literal["group", "layer"] = "layer",
    feat_extract_activation: ActivationType = "gelu",
    layer_norm_eps: float = 1e-5,
    feat_proj_dropout: float = 0.0,
    feat_proj_layer_norm: bool = True,
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
            feat_extract_norm=feat_extract_norm,
            feat_extract_activation=feat_extract_activation,
            layer_norm_eps=layer_norm_eps,
            feat_proj_dropout=feat_proj_dropout,
            feat_proj_layer_norm=feat_proj_layer_norm,
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--key", choices=get_args(PretrainedWavCodecSize), default="librivox")
    parser.add_argument("-c", "--chunk-size", type=int, default=16000 // 10)
    parser.add_argument("-n", "--num-chunks", type=int, default=50 * 10)
    args = parser.parse_args()

    model = pretrained_wav_codec(args.key)
    state: WavCodecState | None = None

    import sounddevice as sd  # type: ignore[import]

    with sd.InputStream(samplerate=16000, channels=1, dtype="float32") as stream:
        for _ in range(args.num_chunks):
            data, overflowed = stream.read(args.chunk_size)
            waveform = torch.from_numpy(data.reshape(1, -1)).float()
            logits, state = model(waveform, state)
            logger.info("Read: %s (overflowed: %s)", logits.argmax(-1).squeeze(0).cpu().tolist(), overflowed)


if __name__ == "__main__":
    # python -m pretrained.wav_codec
    test_wav_codec()
