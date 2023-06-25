"""Defines a pre-trained HiFi-GAN vocoder model.

This vocoder can be used with TTS models that output mel spectrograms to
synthesize audio.

.. code-block:: python

    from pretrained.vocoder import pretrained_vocoder

    vocoder = pretrained_vocoder("hifigan")
"""

from dataclasses import dataclass
from typing import TypeVar, cast

import torch
import torch.nn.functional as F
from ml.core.config import conf_field
from ml.models.lora import SupportedModule as LoraModule, maybe_lora
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.timer import Timer
from torch import Tensor, nn
from torch.nn.utils import remove_weight_norm, weight_norm

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
