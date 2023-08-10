"""Defines a simple API for interacting with Meta's pretrained Encodec model.

This API only supports the causal version of the model, which is better suited
for streaming applications.

.. highlight:: python
.. code-block:: python

    from pretrained.encodec import pretrained_encodec

    model = pretrained_encodec("24khz")
    encoder, decoder = model.get_encoder(), model.get_decoder()

    # Get the tokens for a waveform.
    tokens = encoder.encode(torch.randn(1, 24_000))

    # Reconstructs the waveform from the tokens.
    reconstructed = decoder.decode(tokens)
"""

import argparse
import logging
import math
import warnings
from dataclasses import dataclass
from typing import Literal, Sequence, cast, get_args

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from ml.models.activations import ActivationType, get_activation
from ml.models.codebook import ResidualVectorQuantization, VectorQuantization
from ml.models.norms import NormType, ParametrizationNormType, get_norm_1d, get_parametrization_norm
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer
from torch import Tensor, nn

logger = logging.getLogger(__name__)

PretrainedEncodecSize = Literal["24khz"]


def cast_pretrained_encodec_type(s: str) -> PretrainedEncodecSize:
    if s not in get_args(PretrainedEncodecSize):
        raise KeyError(f"Invalid Enodec type: {s} Expected one of: {get_args(PretrainedEncodecSize)}")
    return cast(PretrainedEncodecSize, s)


PadMode = Literal["reflect", "replicate", "circular", "constant"]


def get_extra_padding_for_conv1d(
    x: Tensor,
    kernel_size: int,
    stride: int,
    padding_total: int = 0,
) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x: Tensor, paddings: tuple[int, int], mode: PadMode = "constant", value: float = 0.0) -> Tensor:
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: Tensor, paddings: tuple[int, int]) -> Tensor:
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConv1d(nn.Module):
    def __init__(
        self,
        conv: nn.Conv1d,
        norm: NormType | ParametrizationNormType,
        *,
        groups: int = 1,
    ) -> None:
        super().__init__()

        param_norm: ParametrizationNormType
        layer_norm: NormType
        if norm in get_args(ParametrizationNormType):
            param_norm = cast(ParametrizationNormType, norm)
            layer_norm = "no_norm"
        else:
            param_norm = "no_norm"
            layer_norm = cast(NormType, norm)

        self.conv = get_parametrization_norm(conv, param_norm)
        self.norm = get_norm_1d(layer_norm, dim=conv.out_channels, groups=groups)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    def __init__(
        self,
        conv: nn.ConvTranspose1d,
        norm: NormType | ParametrizationNormType,
        *,
        groups: int = 1,
    ) -> None:
        super().__init__()

        param_norm: ParametrizationNormType
        layer_norm: NormType
        if norm in get_args(ParametrizationNormType):
            param_norm = cast(ParametrizationNormType, norm)
            layer_norm = "no_norm"
        else:
            param_norm = "no_norm"
            layer_norm = cast(NormType, norm)

        self.convtr = get_parametrization_norm(conv, param_norm)
        self.norm = get_norm_1d(layer_norm, dim=conv.out_channels, groups=groups)

    def forward(self, x: Tensor) -> Tensor:
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConv1d(nn.Module):
    __constants__ = ["stride", "dilation", "kernel_size", "padding_total", "causal", "pad_mode"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: NormType | ParametrizationNormType = "no_norm",
        norm_groups: int = 1,
        pad_mode: PadMode = "reflect",
    ) -> None:
        super().__init__()

        if stride > 1 and dilation > 1:
            warnings.warn(
                "SConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )

        if causal:
            assert norm not in ("group", "group_affine"), "Group norm is not supported for causal convolutions"

        self.conv = NormConv1d(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            norm=norm,
            groups=norm_groups,
        )

        self.stride = self.conv.conv.stride[0]
        self.dilation = self.conv.conv.dilation[0]
        kernel_size = self.conv.conv.kernel_size[0]
        self.kernel_size = (kernel_size - 1) * self.dilation + 1
        self.padding_total = self.kernel_size - self.stride
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x: Tensor) -> Tensor:
        extra_padding = get_extra_padding_for_conv1d(x, self.kernel_size, self.stride, self.padding_total)
        if self.causal:
            x = pad1d(x, (self.padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: NormType | ParametrizationNormType = "no_norm",
        norm_groups: int = 1,
        trim_right_ratio: float = 1.0,
    ) -> None:
        super().__init__()

        if causal:
            assert norm not in ("group", "group_affine"), "Group norm is not supported for causal convolutions"

        self.causal = causal
        self.trim_right_ratio = trim_right_ratio

        if not (self.causal or self.trim_right_ratio == 1.0):
            raise ValueError("`trim_right_ratio` != 1.0 only makes sense for causal convolutions")
        if self.trim_right_ratio < 0.0 or self.trim_right_ratio > 1.0:
            raise ValueError("`trim_right_ratio` must be in [0.0, 1.0]")

        self.convtr = NormConvTranspose1d(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
            ),
            norm=norm,
            groups=norm_groups,
        )

    def forward(self, x: Tensor) -> Tensor:
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride
        y = self.convtr(x)
        if self.causal:
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y


class SLSTM(nn.Module):
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True) -> None:
        super().__init__()

        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x.float())
        y = y.to(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y


class SEANetResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_sizes: Sequence[int] = [3, 1],
        dilations: Sequence[int] = [1, 1],
        activation: ActivationType = "elu",
        norm: NormType | ParametrizationNormType = "weight",
        causal: bool = False,
        pad_mode: PadMode = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ) -> None:
        super().__init__()

        assert len(kernel_sizes) == len(dilations), "Number of kernel sizes should match number of dilations"
        hidden = dim // compress

        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                get_activation(activation, inplace=False),
                SConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.Sequential(*block)

        self.shortcut = (
            nn.Identity()
            if true_skip
            else SConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                causal=causal,
                pad_mode=pad_mode,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: Sequence[int] = [8, 5, 4, 2],
        activation: ActivationType = "elu",
        norm: NormType | ParametrizationNormType = "weight",
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: PadMode = "reflect",
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        mult = 1
        model: list[nn.Module] = [
            SConv1d(
                channels,
                mult * n_filters,
                kernel_size,
                norm=norm,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        # Downsample to raw audio scale
        for ratio in self.ratios:
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=norm,
                        activation=activation,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            # Add downsampling layers
            model += [
                get_activation(activation, inplace=False),
                SConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]

        model += [
            get_activation(activation, inplace=False),
            SConv1d(
                mult * n_filters,
                dimension,
                last_kernel_size,
                norm=norm,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class SEANetDecoder(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: Sequence[int] = [8, 5, 4, 2],
        activation: ActivationType = "elu",
        final_activation: ActivationType = "no_act",
        norm: NormType | ParametrizationNormType = "weight",
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: PadMode = "reflect",
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
        trim_right_ratio: float = 1.0,
    ) -> None:
        super().__init__()

        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        mult = int(2 ** len(self.ratios))
        model: list[nn.Module] = [
            SConv1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm=norm,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]

        # Upsample to raw audio scale.
        for ratio in self.ratios:
            # Add upsampling layers.
            model += [
                get_activation(activation, inplace=False),
                SConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            ]

            # Add residual layers.
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        norm=norm,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            mult //= 2

        # Add final layers.
        model += [
            get_activation(activation, inplace=False),
            SConv1d(
                n_filters,
                channels,
                last_kernel_size,
                norm=norm,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        # Add optional final activation to decoder.
        if final_activation is not None:
            model += [get_activation(activation, inplace=False)]
        self.model = nn.Sequential(*model)

    def forward(self, z: Tensor) -> Tensor:
        y = self.model(z)
        return y


def _encode(x: Tensor, encoder: SEANetEncoder, quantizer: ResidualVectorQuantization) -> Tensor:
    _, channels, _ = x.shape
    assert channels > 0 and channels <= 2
    emb = encoder(x)
    codes = quantizer.encode(emb)
    codes = codes.transpose(0, 1)
    return codes


def _decode(tokens: Tensor, decoder: SEANetDecoder, quantizer: ResidualVectorQuantization) -> Tensor:
    codes = tokens.transpose(0, 1)
    emb = quantizer.decode(codes)
    out = decoder(emb)
    return out


class Encodec(nn.Module):
    """EnCodec model operating on the raw waveform.

    Parameters:
        encoder: Encoder network.
        decoder: Decoder network.
        quantizer: The residual quantizer module.
        channels: Number of audio channels.
        sample_rate: The sample rate of the input audio.
    """

    def __init__(
        self,
        encoder: SEANetEncoder,
        decoder: SEANetDecoder,
        quantizer: ResidualVectorQuantization,
        channels: int = 1,
        sample_rate: int | None = None,
    ) -> None:
        super().__init__()

        self.bandwidth: float | None = None
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.channels = channels
        self.sample_rate = sample_rate

        quantizer_bins = quantizer.layers[0].codebook_size
        self.bits_per_codebook = int(math.log2(quantizer_bins))
        if 2**self.bits_per_codebook != quantizer_bins:
            raise ValueError(f"Invalid number of quantizer bins: {quantizer_bins}. Must be a power of 2.")

    def get_encoder(self) -> "Encoder":
        return Encoder(self)

    def get_decoder(self) -> "Decoder":
        return Decoder(self)

    def encode(self, waveform: Tensor) -> Tensor:
        return _encode(waveform, encoder=self.encoder, quantizer=self.quantizer)

    def decode(self, tokens: Tensor) -> Tensor:
        return _decode(tokens, decoder=self.decoder, quantizer=self.quantizer)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder(x)
        x, _, vq_losses, _ = self.quantizer(x)
        x = self.decoder(x)
        return x, vq_losses


class Encoder(nn.Module):
    def __init__(self, encodec: Encodec) -> None:
        super().__init__()

        self.encoder = encodec.encoder
        self.quantizer = encodec.quantizer

    def encode(self, waveform: Tensor) -> Tensor:
        return _encode(waveform, encoder=self.encoder, quantizer=self.quantizer)

    def forward(self, waveform: Tensor) -> Tensor:
        return self.encode(waveform)


class Decoder(nn.Module):
    def __init__(self, encodec: Encodec) -> None:
        super().__init__()

        self.decoder = encodec.decoder
        self.quantizer = encodec.quantizer

    def decode(self, tokens: Tensor) -> Tensor:
        return _decode(tokens, decoder=self.decoder, quantizer=self.quantizer)

    def forward(self, waveform: Tensor) -> Tensor:
        return self.decode(waveform)


@dataclass
class EncodecConfig:
    num_quantizers: int
    channels: int
    causal: bool
    norm: NormType | ParametrizationNormType
    sample_rate: int | None = None


def _load_pretrained_encodec(
    size: PretrainedEncodecSize,
    ckpt_url: str,
    sha256: str,
    config: EncodecConfig,
    load_weights: bool = True,
) -> Encodec:
    encoder = SEANetEncoder(channels=config.channels, norm=config.norm, causal=config.causal)
    decoder = SEANetDecoder(channels=config.channels, norm=config.norm, causal=config.causal)
    quantizer = ResidualVectorQuantization(
        VectorQuantization(dim=encoder.dimension, codebook_size=1024),
        num_quantizers=config.num_quantizers,
    )
    model = Encodec(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        channels=config.channels,
        sample_rate=config.sample_rate,
    )

    # Loads the model weights.
    if load_weights:
        model_fname = f"{size}.bin"

        with Timer("downloading checkpoint"):
            model_path = ensure_downloaded(ckpt_url, "hubert", model_fname, sha256=sha256)

        def change_prefix(s: str, a: str, b: str) -> str:
            if s.startswith(a):
                return b + s[len(a) :]
            return s

        with Timer("loading checkpoint", spinner=True):
            ckpt = torch.load(model_path, map_location="cpu")
            ckpt = {change_prefix(k, "quantizer.vq.", "quantizer."): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt)

    return model


def pretrained_encodec(size: str | PretrainedEncodecSize, load_weights: bool = True) -> Encodec:
    match size:
        case "24khz":
            return _load_pretrained_encodec(
                cast_pretrained_encodec_type(size),
                ckpt_url="https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th",
                sha256="d7cc33bcf1aad7f2dad9836f36431530744abeace3ca033005e3290ed4fa47bf",
                config=EncodecConfig(
                    num_quantizers=32,
                    channels=1,
                    causal=True,
                    norm="weight",
                    sample_rate=24000,
                ),
                load_weights=load_weights,
            )
        case _:
            raise NotImplementedError(f"Invalid size: {size}")


def test_encodec_adhoc() -> None:
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=str, choices=get_args(PretrainedEncodecSize))
    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument("output_file", type=str, help="Path to output audio file")
    args = parser.parse_args()

    # Loads the encoder and decoder.
    model = pretrained_encodec(args.size)
    encoder, decoder = model.get_encoder(), model.get_decoder()

    # Loads the audio file.
    audio, sr = torchaudio.load(args.input_file)
    audio = audio[None, :, : sr * 10]
    if model.sample_rate is not None and sr != model.sample_rate:
        audio = torchaudio.functional.resample(audio, sr, model.sample_rate)
        sr = model.sample_rate

    # Runs the codec.
    tokens = encoder.encode(audio)
    reconstructed_audio = decoder.decode(tokens)

    # Saves the audio.
    torchaudio.save(args.output_file, reconstructed_audio[0], sr)

    logger.info("Saved %s", args.output_file)


if __name__ == "__main__":
    # python -m pretrained.encodec
    test_encodec_adhoc()
