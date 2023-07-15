"""Implementation of the Demucs model architecture.

From the paper `Real Time Speech Enhancement in the Waveform Domain
<https://arxiv.org/abs/2006.12847>`_. The paper has a project page
`here <https://github.com/facebookresearch/denoiser>`_.

This model is a relatively straight-forward autoencoder, similar to a UNet but
with an RNN in between. The original model was trained to do denoising, which
makes sense for this particular model since it simply requires removing some
part of the input waveform.
"""

import functools
import math
import time
from typing import cast

import ml.api as ml
import torch
import torch.nn.functional as F
from ml.utils.device.auto import detect_device
from ml.utils.device.base import base_device
from torch import Tensor, nn


def sinc(t: Tensor) -> Tensor:
    return torch.where(t == 0, torch.tensor(1.0, device=t.device, dtype=t.dtype), torch.sin(t) / t)


@functools.lru_cache()
def kernel_upsample2(device: torch.device, dtype: torch.dtype, zeros: int = 56) -> Tensor:
    win = torch.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel.to(device, dtype)


def upsample2(x: Tensor, zeros: int = 56) -> Tensor:
    *other, time = x.shape
    kernel = kernel_upsample2(x.device, x.dtype, zeros)
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
    y = torch.stack([x, out], dim=-1)
    return y.view(*other, -1)


@functools.lru_cache()
def kernel_downsample2(device: torch.device, dtype: torch.dtype, zeros: int = 56) -> Tensor:
    win = torch.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel.to(device, dtype)


def downsample2(x: Tensor, zeros: int = 56) -> Tensor:
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape
    kernel = kernel_downsample2(x.device, x.dtype, zeros)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(*other, time)
    return out.view(*other, -1).mul(0.5)


def fast_conv(conv: nn.Conv1d | nn.ConvTranspose1d, x: Tensor) -> Tensor:
    batch, in_channels, length = x.shape
    weight, bias = conv.weight, conv.bias
    out_channels, in_channels, kernel = weight.shape
    assert batch == 1
    if bias is None:
        out = conv(x)
    elif kernel == 1:
        x = x.view(in_channels, length)
        out = torch.addmm(bias.view(-1, 1), weight.view(out_channels, in_channels), x)
    elif length == kernel:
        x = x.view(in_channels * kernel, 1)
        out = torch.addmm(bias.view(-1, 1), weight.view(out_channels, in_channels * kernel), x)
    else:
        out = conv(x)
    return out.view(batch, out_channels, -1)


def rescale_conv(conv: nn.Conv1d | nn.ConvTranspose1d, reference: float) -> None:
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module: nn.Module, reference: float) -> None:
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class RNN(nn.Module):
    def __init__(self, dim: int, layers: int = 2, bi: bool = True) -> None:
        super().__init__()

        self.lstm = nn.LSTM(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x: Tensor, hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        act: ml.ActivationType = "relu",
    ) -> None:
        super().__init__()

        self.conv_a = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)
        self.conv_b = nn.Conv1d(out_channels, out_channels * 2, 1)
        self.act = ml.get_activation(act)
        self.glu = nn.GLU(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_a(x)
        x = self.act(x)
        x = self.conv_b(x)
        x = self.glu(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        act: ml.ActivationType = "relu",
    ) -> None:
        super().__init__()

        self.conv_a = nn.Conv1d(in_channels, in_channels * 2, 1, stride=1)
        self.conv_b = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        self.act = ml.get_activation(act)
        self.glu = nn.GLU(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.glu(self.conv_a(x))
        x = self.act(self.conv_b(x))
        return x


class Demucs(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int = 48,
        depth: int = 5,
        kernel_size: int = 8,
        stride: int = 4,
        causal: bool = True,
        resample: int = 4,
        growth: float = 2,
        max_hidden: int = 10_000,
        normalize: bool = True,
        rescale: float = 0.1,
        floor: float = 1e-3,
        sample_rate: int = 16_000,
    ) -> None:
        """Demucs speech enhancement model.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden: Number of initial hidden channels.
            depth: Number of layers.
            kernel_size: Kernel size for each layer.
            stride: Stride for each layer.
            causal: If false, uses BiLSTM instead of LSTM.
            resample: Amount of resampling to apply to the input/output.
                Can be one of 1, 2 or 4.
            growth: Number of channels is multiplied by this for every layer.
            max_hidden: Maximum number of channels. Can be useful to
                control the size/speed of the model.
            normalize: If true, normalize the input.
            rescale: Controls custom weight initialization.
            floor: Floor value for normalization.
            sample_rate: Sample rate used for training the model.
        """
        super().__init__()

        if resample not in [1, 2, 4]:
            raise ValueError("Resample must be one of 1, 2 or 4")

        # Model parameters.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.growth = growth
        self.max_hidden = max_hidden
        self.rescale = rescale

        # Used during training.
        self.resample = resample
        self.normalize = normalize
        self.floor = floor
        self.sample_rate = sample_rate

        encoders: list[Encoder] = []
        decoders: list[Decoder] = []

        for index in range(depth):
            encoder = Encoder(in_channels, hidden, kernel_size, stride)
            decoder = Decoder(hidden, out_channels, kernel_size, stride, act="relu" if index > 0 else "no_act")
            encoders.append(encoder)
            decoders.append(decoder)
            in_channels = hidden
            out_channels = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = RNN(in_channels, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

        self.encoders = cast(list[Encoder], nn.ModuleList(encoders))
        self.decoders = cast(list[Decoder], nn.ModuleList(decoders[::-1]))

    def valid_length(self, length: int) -> int:
        """Returns the nearest valid length to use with the model.

        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.

        Args:
            length: Length of the input.

        Returns:
            The nearest valid length.
        """
        length = math.ceil(length * self.resample)
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self) -> int:
        return self.stride**self.depth // self.resample

    def forward(self, mix: Tensor) -> Tensor:
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        std: Tensor | None = None
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for encode in self.encoders:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm.forward(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoders:
            skip = skips.pop(-1)
            x = x + skip[..., : x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        if std is not None:
            x = x * std
        return x

    def streamer(
        self,
        *,
        dry: float = 0.0,
        num_frames: int = 1,
        resample_lookahead: int = 64,
        resample_buffer: int = 256,
        device: base_device | None = None,
    ) -> "DemucsStreamer":
        """Gets a streamer for the current model.

        Args:
            dry: Percentage of the unaltered signal to preserve (0 to 1).
            num_frames: Number of frames to process at once. Higher values
                will increase overall latency but improve the real time factor.
            resample_lookahead: Extra lookahead used for the resampling.
            resample_buffer: Size of the buffer of previous inputs/outputs
                kept for resampling.
            device: The device to use for predictions. If `None`, will use the
                device returned by detect_device().

        Returns:
            A streamer for streaming from the current model.
        """
        return DemucsStreamer(
            self,
            dry=dry,
            num_frames=num_frames,
            resample_lookahead=resample_lookahead,
            resample_buffer=resample_buffer,
            device=device,
        )


class DemucsStreamer:
    def __init__(
        self,
        demucs: Demucs,
        dry: float = 0.0,
        num_frames: int = 1,
        resample_lookahead: int = 64,
        resample_buffer: int = 256,
        device: base_device | None = None,
    ) -> None:
        self.device = detect_device() if device is None else device
        self.demucs = demucs
        self.device.module_to(self.demucs)
        self.lstm_state: Tensor | None = None
        self.conv_state: list[Tensor] | None = None
        self.dry = dry
        self.resample_lookahead = resample_lookahead
        resample_buffer = min(demucs.total_stride, resample_buffer)
        self.resample_buffer = resample_buffer
        self.frame_length = demucs.valid_length(1) + demucs.total_stride * (num_frames - 1)
        self.total_length = self.frame_length + self.resample_lookahead
        self.stride = demucs.total_stride * num_frames
        self.resample_in = self.device.tensor_to(torch.zeros(demucs.in_channels, resample_buffer))
        self.resample_out = self.device.tensor_to(torch.zeros(demucs.in_channels, resample_buffer))

        self.frames = 0
        self.total_time = 0.0
        self.variance = 0.0
        self.pending = self.device.tensor_to(torch.zeros(demucs.in_channels, 0))

    def reset_time_per_frame(self) -> None:
        self.total_time = 0
        self.frames = 0

    @property
    def time_per_frame(self) -> float:
        return self.total_time / self.frames

    def flush(self) -> Tensor:
        self.lstm_state = None
        self.conv_state = None
        pending_length = self.pending.shape[1]
        padding = torch.zeros(self.demucs.in_channels, self.total_length, device=self.pending.device)
        out = self.feed(padding)
        return out[:, :pending_length]

    def feed(self, wav: Tensor) -> Tensor:
        begin = time.time()

        if wav.dim() != 2:
            raise ValueError("input wav should be two dimensional.")
        in_channels, _ = wav.shape
        if in_channels != self.demucs.in_channels:
            raise ValueError(f"Expected {self.demucs.in_channels} channels, got {in_channels}")

        self.pending = torch.cat([self.pending, wav], dim=1)
        outs: list[Tensor] = []
        while self.pending.shape[1] >= self.total_length:
            self.frames += 1
            frame = self.pending[:, : self.total_length]
            dry_signal = frame[:, : self.stride]
            if self.demucs.normalize:
                mono = frame.mean(0)
                variance = (mono**2).mean()
                self.variance = variance / self.frames + (1 - 1 / self.frames) * self.variance
                frame = frame / (self.demucs.floor + math.sqrt(self.variance))
            padded_frame = torch.cat([self.resample_in, frame], dim=-1)
            self.resample_in[:] = frame[:, self.stride - self.resample_buffer : self.stride]
            frame = padded_frame

            if self.demucs.resample == 4:
                frame = upsample2(upsample2(frame))
            elif self.demucs.resample == 2:
                frame = upsample2(frame)
            frame = frame[:, self.demucs.resample * self.resample_buffer :]
            frame = frame[:, : self.demucs.resample * self.frame_length]

            out, extra = self._separate_frame(frame)
            padded_out = torch.cat([self.resample_out, out, extra], 1)
            self.resample_out[:] = out[:, -self.resample_buffer :]
            if self.demucs.resample == 4:
                out = downsample2(downsample2(padded_out))
            elif self.demucs.resample == 2:
                out = downsample2(padded_out)
            else:
                out = padded_out

            out = out[:, self.resample_buffer // self.demucs.resample :]
            out = out[:, : self.stride]

            if self.demucs.normalize:
                out *= math.sqrt(self.variance)
            out = self.dry * dry_signal + (1 - self.dry) * out
            outs.append(out)
            self.pending = self.pending[:, self.stride :]

        self.total_time += time.time() - begin
        if outs:
            out = torch.cat(outs, 1)
        else:
            out = torch.zeros(in_channels, 0, device=wav.device)
        return out

    def _separate_frame(self, frame: Tensor) -> tuple[Tensor, Tensor]:
        skips: list[Tensor] = []
        next_state: list[Tensor] = []
        stride = self.stride * self.demucs.resample
        x = frame[None]
        for idx, encode in enumerate(self.demucs.encoders):
            stride //= self.demucs.stride
            length = x.shape[2]
            if idx == self.demucs.depth - 1:
                x = fast_conv(encode.conv_a, x)
                x = encode.act(x)
                x = fast_conv(encode.conv_b, x)
                x = encode.glu(x)
            else:
                if not_first := self.conv_state is not None:
                    prev = self.conv_state.pop(0)
                    prev = prev[..., stride:]
                    tgt = (length - self.demucs.kernel_size) // self.demucs.stride + 1
                    missing = tgt - prev.shape[-1]
                    offset = length - self.demucs.kernel_size - self.demucs.stride * (missing - 1)
                    x = x[..., offset:]
                x = encode.act(encode.conv_a(x))
                x = fast_conv(encode.conv_b, x)
                x = encode.glu(x)
                if not_first:
                    x = torch.cat([prev, x], -1)
                next_state.append(x)
            skips.append(x)

        x = x.permute(2, 0, 1)
        x, self.lstm_state = self.demucs.lstm.forward(x, self.lstm_state)
        x = x.permute(1, 2, 0)

        # In the following, x contains only correct samples, i.e. the one for
        # which each time position is covered by two window of the upper layer.
        # extra contains extra samples to the right, and is used only as a
        # better padding for the online resampling.
        extra: Tensor | None = None
        for idx, decode in enumerate(self.demucs.decoders):
            skip = skips.pop(-1)
            x += skip[..., : x.shape[-1]]
            x = fast_conv(decode.conv_a, x)
            x = decode.glu(x)

            if extra is not None:
                skip = skip[..., x.shape[-1] :]
                extra += skip[..., : extra.shape[-1]]
                extra = decode.conv_b(decode.glu(decode.conv_a(extra)))
            x = decode.conv_b(x)
            next_state.append(x[..., -self.demucs.stride :] - cast(Tensor, decode.conv_b.bias).view(-1, 1))
            if extra is None:
                extra = x[..., -self.demucs.stride :]
            else:
                extra[..., : self.demucs.stride] += next_state[-1]
            x = x[..., : -self.demucs.stride]

            if self.conv_state is not None:
                prev = self.conv_state.pop(0)
                x[..., : self.demucs.stride] += prev
            if idx != self.demucs.depth - 1:
                x = decode.act(x)
                extra = decode.act(extra)
        self.conv_state = next_state
        assert extra is not None, "Extra is None!"
        return x[0], extra[0]
