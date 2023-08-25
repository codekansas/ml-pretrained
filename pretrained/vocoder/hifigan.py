"""Defines a pre-trained HiFi-GAN vocoder model.

This vocoder can be used with TTS models that output mel spectrograms to
synthesize audio.

.. code-block:: python

    from pretrained.vocoder import pretrained_vocoder

    vocoder = pretrained_vocoder("hifigan")
"""

import argparse
import logging
from typing import Sequence, TypeVar, cast

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from ml.models.modules import StreamingConv1d, StreamingConvTranspose1d, streaming_add
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer
from torch import Tensor, nn
from torch.nn.utils import remove_weight_norm, weight_norm

logger = logging.getLogger(__name__)

HIFIGAN_CKPT_URL = "https://huggingface.co/jaketae/hifigan-lj-v1/resolve/main/pytorch_model.bin"

T = TypeVar("T")


def get(x: Sequence[T] | None, i: int) -> T | None:
    return None if x is None else x[i]


def init_hifigan_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, StreamingConv1d, StreamingConvTranspose1d)):
        m.weight.data.normal_(mean, std)


StreamingConvState = tuple[Tensor, int]
StreamingAddState = tuple[Tensor, Tensor]
ResBlockState = list[tuple[StreamingConvState, StreamingConvState, StreamingAddState]]
HiFiGANState = tuple[
    StreamingConvState,
    list[StreamingConvState],
    list[ResBlockState],
    list[StreamingAddState],
    StreamingConvState,
]


class ResBlock(nn.Module):
    __constants__ = ["lrelu_slope"]

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
    ) -> None:
        super().__init__()

        def get_padding(kernel_size: int, dilation: int = 1) -> int:
            return (kernel_size * dilation - dilation) // 2

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    StreamingConv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    StreamingConv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    StreamingConv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_hifigan_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    StreamingConv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    StreamingConv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    StreamingConv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_hifigan_weights)

        self.lrelu_slope = lrelu_slope

    def forward(self, x: Tensor, state: ResBlockState | None) -> tuple[Tensor, ResBlockState]:
        state_out: ResBlockState = []
        for i, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            state_in_i = get(state, i)
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt, s1 = c1(xt, get(state_in_i, 0))
            xt = F.leaky_relu(xt, self.lrelu_slope)
            xt, s2 = c2(xt, get(state_in_i, 1))
            x, sa = streaming_add(xt, x, get(state_in_i, 2))
            state_out.append((s1, s2, sa))
        return x, state_out

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class HiFiGAN(nn.Module):
    """Defines a HiFi-GAN model.

    Parameters:
        resblock_kernel_sizes: The kernel sizes of the ResBlocks.
        resblock_dilation_sizes: The dilation sizes of the ResBlocks.
        upsample_rates: The upsample rates of each layer.
        upsample_initial_channel: The initial channel of the upsampling layers.
        upsample_kernel_sizes: The kernel sizes of the upsampling layers.
        model_in_dim: The input dimension of the model.
        sampling_rate: The sampling rate of the model.
        lrelu_slope: The slope of the leaky ReLU.
    """

    def __init__(
        self,
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilation_sizes: list[tuple[int, int, int]] = [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        upsample_rates: list[int] = [8, 8, 2, 2],
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: list[int] = [16, 16, 4, 4],
        model_in_dim: int = 80,
        sampling_rate: int = 22050,
        lrelu_slope: float = 0.1,
    ) -> None:
        super().__init__()

        self.model_in_dim = model_in_dim
        self.sampling_rate = sampling_rate
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.lrelu_slope = lrelu_slope
        conv_pre = StreamingConv1d(model_in_dim, upsample_initial_channel, kernel_size=7, stride=1, padding=3)
        self.conv_pre = weight_norm(conv_pre)

        assert len(upsample_rates) == len(upsample_kernel_sizes)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            module = StreamingConvTranspose1d(
                upsample_initial_channel // (2**i),
                upsample_initial_channel // (2 ** (i + 1)),
                kernel_size=k,
                stride=u,
                # padding=(k - u) // 2,
            )
            self.ups.append(weight_norm(module))

        self.resblocks = cast(list[ResBlock], nn.ModuleList())
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d, lrelu_slope))

        self.conv_post = weight_norm(StreamingConv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_hifigan_weights)
        self.conv_post.apply(init_hifigan_weights)

    def forward(self, x: Tensor, state: HiFiGANState | None = None) -> tuple[Tensor, HiFiGANState]:
        x, pre_s = self.conv_pre(x, get(state, 0))
        up_s_in = get(state, 1)
        up_s_out: list[StreamingConvState] = []
        down_s_in = get(state, 2)
        down_s_out: list[list[StreamingConvState]] = []
        sa_in = get(state, 3)
        sa_out: list[list[StreamingAddState]] = []
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, self.lrelu_slope)
            x, up_s = up(x, get(up_s_in, i))
            up_s_out.append(up_s)
            xs = None
            down_s_in_i, down_s_out_i = get(down_s_in, i), []
            sa_in_i = get(sa_in, i)
            sa_out_i: list[StreamingAddState] = []
            for j in range(self.num_kernels):
                down_s_in_ij = get(down_s_in_i, j)
                sa_in_ij = get(sa_in_i, j - 1)
                xs_i, down_s_out_ij = self.resblocks[i * self.num_kernels + j](x, down_s_in_ij)
                xs, sa_i = (xs_i, None) if xs is None else streaming_add(xs, xs_i, sa_in_ij)
                if sa_i is not None:
                    sa_out_i.append(sa_i)
                down_s_out_i.append(down_s_out_ij)
            down_s_out.append(down_s_out_i)
            sa_out.append(sa_out_i)
            assert xs is not None
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x, post_s = self.conv_post(x, get(state, 4))
        x = torch.tanh(x)

        return x, (pre_s, up_s_out, down_s_out, sa_out, post_s)

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def _load_hifigan_weights(
    model: HiFiGAN,
    url: str,
    load_weights: bool = True,
    device: torch.device | None = None,
) -> None:
    if not load_weights:
        return model

    with Timer("downloading checkpoint"):
        model_path = ensure_downloaded(url, "hifigan", "weights_hifigan.pth")

    with Timer("loading checkpoint", spinner=True):
        if device is None:
            device = torch.device("cpu")
        ckpt = torch.load(model_path, map_location=device)
        model.to(device)
        model.load_state_dict(ckpt)

    return model


def pretrained_hifigan(*, pretrained: bool = True) -> HiFiGAN:
    """Loads the pretrained HiFi-GAN model.

    Args:
        pretrained: Whether to load the pretrained weights.

    Returns:
        The pretrained HiFi-GAN model.
    """
    with Timer("initializing model", spinner=True):
        model = HiFiGAN()
    _load_hifigan_weights(model, HIFIGAN_CKPT_URL, pretrained)
    return model


class AudioToHifiGanMels(nn.Module):
    """Defines a module to convert from a waveform to the mels used by HiFi-GAN.

    The default parameters should be kept the same for pre-trained models.

    Parameters:
        sampling_rate: The sampling rate of the audio.
        num_mels: The number of mel bins.
        n_fft: The number of FFT bins.
        win_size: The window size.
        hop_size: The hop size.
        fmin: The minimum frequency.
        fmax: The maximum frequency.
    """

    __constants__ = ["sampling_rate", "num_mels", "n_fft", "win_size", "hop_size", "fmin", "fmax"]

    def __init__(
        self,
        sampling_rate: int = 22050,
        num_mels: int = 80,
        n_fft: int = 1024,
        win_size: int = 1024,
        hop_size: int = 256,
        fmin: int = 0,
        fmax: int = 8000,
    ) -> None:
        super().__init__()

        self.sampling_rate = sampling_rate
        self.num_mels = num_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax

        # try:
        #     from librosa.filters import mel as librosa_mel_fn  # type: ignore[import]
        # except ImportError:
        #     raise ImportError("Please install librosa to use AudioToHifiGanMels")

        # mel_librosa = librosa_mel_fn(
        #     sr=sampling_rate,
        #     n_fft=n_fft,
        #     n_mels=num_mels,
        #     fmin=fmin,
        #     fmax=fmax,
        # )
        # mel = torch.from_numpy(mel_librosa).float().T

        mel = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=fmin,
            f_max=fmax,
            n_mels=num_mels,
            sample_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

        self.register_buffer("mel_basis", mel)
        self.register_buffer("hann_window", torch.hann_window(win_size))

    def _dynamic_range_compression(self, x: np.ndarray, c: float = 1.0, clip_val: float = 1e-5) -> np.ndarray:
        return np.log(np.clip(x, a_min=clip_val, a_max=None) * c)

    def _dynamic_range_decompression(self, x: np.ndarray, c: float = 1.0) -> np.ndarray:
        return np.exp(x) / c

    def _dynamic_range_compression_torch(self, x: Tensor, c: float = 1.0, clip_val: float = 1e-5) -> Tensor:
        return torch.log(torch.clamp(x, min=clip_val) * c)

    def _dynamic_range_decompression_torch(self, x: Tensor, c: float = 1.0) -> Tensor:
        return torch.exp(x) / c

    def _spectral_normalize_torch(self, magnitudes: Tensor) -> Tensor:
        output = self._dynamic_range_compression_torch(magnitudes)
        return output

    def _spectral_de_normalize_torch(self, magnitudes: Tensor) -> Tensor:
        output = self._dynamic_range_decompression_torch(magnitudes)
        return output

    mel_basis: Tensor
    hann_window: Tensor

    def wav_to_mels(
        self,
        y: Tensor,
        center: bool = False,
    ) -> Tensor:
        ymin, ymax = torch.min(y), torch.max(y)
        if ymin < -1.0:
            logger.warning("min value is %.2g", ymin)
        if ymax > 1.0:
            logger.warning("max value is %.2g", ymax)

        pad = int((self.n_fft - self.hop_size) / 2)
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect")
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
        spec = torch.einsum("bct,cm->bmt", spec, self.mel_basis)
        spec = self._spectral_normalize_torch(spec)

        return spec


def test_mel_to_audio_adhoc() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Runs adhoc test of mel to audio conversion")
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

    # Loads the HiFi-GAN model.
    model = pretrained_hifigan(pretrained=True)

    # Converts the audio to mels.
    audio_to_mels = AudioToHifiGanMels()
    mels = audio_to_mels.wav_to_mels(audio)

    # Converts the mels back to audio.
    audio, _ = model(mels)
    audio = audio.squeeze(0)

    # Saves the audio.
    torchaudio.save(args.output_file, audio, 22050)

    logger.info("Saved %s", args.output_file)


if __name__ == "__main__":
    # python -m pretrained.vocoder.hifigan
    test_mel_to_audio_adhoc()
