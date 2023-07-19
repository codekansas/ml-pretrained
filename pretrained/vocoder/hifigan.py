"""Defines a pre-trained HiFi-GAN vocoder model.

This vocoder can be used with TTS models that output mel spectrograms to
synthesize audio.

.. code-block:: python

    from pretrained.vocoder import pretrained_vocoder

    vocoder = pretrained_vocoder("hifigan")
"""

import argparse
import logging
from dataclasses import dataclass
from typing import TypeVar, cast

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from ml.core.config import conf_field
from ml.models.lora import SupportedModule as LoraModule, maybe_lora
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer
from torch import Tensor, nn
from torch.nn.utils import remove_weight_norm, weight_norm

logger = logging.getLogger(__name__)

HIFIGAN_CKPT_URL = "https://huggingface.co/jaketae/hifigan-lj-v1/resolve/main/pytorch_model.bin"


@dataclass
class HiFiGANConfig:
    resblock_kernel_sizes: list[int] = conf_field([3, 7, 11], help="Kernel sizes of ResBlock.")
    resblock_dilation_sizes: list[tuple[int, int, int]] = conf_field(
        [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        help="Dilation sizes of ResBlock.",
    )
    upsample_rates: list[int] = conf_field([8, 8, 2, 2], help="Upsample rates of each layer.")
    upsample_initial_channel: int = conf_field(512, help="Initial channel of upsampling layers.")
    upsample_kernel_sizes: list[int] = conf_field([16, 16, 4, 4], help="Kernel sizes of upsampling layers.")
    model_in_dim: int = conf_field(80, help="Input dimension of model.")
    sampling_rate: int = conf_field(22050, help="Sampling rate of model.")
    lrelu_slope: float = conf_field(0.1, help="Slope of leaky relu.")
    lora_rank: int | None = conf_field(None, help="LoRA rank")


def init_hifigan_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.normal_(mean, std)


T_module = TypeVar("T_module", bound=LoraModule)


def lora_weight_norm(module: T_module, lora_rank: int | None) -> T_module:
    return weight_norm(maybe_lora(module, r=lora_rank))


class ResBlock(nn.Module):
    __constants__ = ["lrelu_slope"]

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
        lora_rank: int | None = None,
    ) -> None:
        super().__init__()

        def get_padding(kernel_size: int, dilation: int = 1) -> int:
            return (kernel_size * dilation - dilation) // 2

        self.convs1 = nn.ModuleList(
            [
                lora_weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    ),
                    lora_rank,
                ),
                lora_weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    ),
                    lora_rank,
                ),
                lora_weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    ),
                    lora_rank,
                ),
            ]
        )
        self.convs1.apply(init_hifigan_weights)

        self.convs2 = nn.ModuleList(
            [
                lora_weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    ),
                    lora_rank,
                ),
                lora_weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    ),
                    lora_rank,
                ),
                lora_weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    ),
                    lora_rank,
                ),
            ]
        )
        self.convs2.apply(init_hifigan_weights)

        self.lrelu_slope = lrelu_slope

    def forward(self, x: Tensor) -> Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class HiFiGAN(nn.Module):
    def __init__(self, config: HiFiGANConfig) -> None:
        super().__init__()

        self.sampling_rate = config.sampling_rate
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.lrelu_slope = config.lrelu_slope
        conv_pre = nn.Conv1d(config.model_in_dim, config.upsample_initial_channel, 7, 1, padding=3)
        self.conv_pre = lora_weight_norm(conv_pre, config.lora_rank)

        assert len(config.upsample_rates) == len(config.upsample_kernel_sizes)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            module = nn.ConvTranspose1d(
                config.upsample_initial_channel // (2**i),
                config.upsample_initial_channel // (2 ** (i + 1)),
                k,
                u,
                padding=(k - u) // 2,
            )
            self.ups.append(lora_weight_norm(module, config.lora_rank))

        self.resblocks = cast(list[ResBlock], nn.ModuleList())
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d, config.lrelu_slope, config.lora_rank))

        self.conv_post = lora_weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3), config.lora_rank)
        self.ups.apply(init_hifigan_weights)
        self.conv_post.apply(init_hifigan_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = up(x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            assert xs is not None
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def infer(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def pretrained_hifigan(
    *,
    pretrained: bool = True,
    lora_rank: int | None = None,
    device: torch.device | None = None,
) -> HiFiGAN:
    """Loads the pretrained HiFi-GAN model.

    Args:
        pretrained: Whether to load the pretrained weights.
        lora_rank: The LoRA rank to use, if LoRA is desired.
        device: The device to load the weights onto.

    Returns:
        The pretrained HiFi-GAN model.
    """
    config = HiFiGANConfig(lora_rank=lora_rank)

    if not pretrained:
        return HiFiGAN(config)

    # Can't initialize empty weights because of weight norm.
    # with Timer("initializing model", spinner=True), init_empty_weights():
    with Timer("initializing model", spinner=True):
        model = HiFiGAN(config)

    with Timer("downloading checkpoint"):
        model_path = ensure_downloaded(HIFIGAN_CKPT_URL, "hifigan", "weights_hifigan.pth")

    with Timer("loading checkpoint", spinner=True):
        if device is None:
            device = torch.device("cpu")
        ckpt = torch.load(model_path, map_location=device)
        model.to(device)
        model.load_state_dict(ckpt)

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
    audio = model(mels).squeeze(0)

    # Saves the audio.
    torchaudio.save(args.output_file, audio, 22050)

    logger.info("Saved %s", args.output_file)


if __name__ == "__main__":
    # python -m pretrained.vocoder.hifigan
    test_mel_to_audio_adhoc()
