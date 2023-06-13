"""Defines a pre-trained WaveGlow vocoder model.

This vocoder can be used with TTS models that output mel spectrograms to
synthesize audio.

.. code-block:: python

    from pretrained.vocoder import pretrained_vocoder

    vocoder = pretrained_vocoder("waveglow")
"""

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from ml.core.config import conf_field
from ml.models.lora import maybe_lora
from ml.utils.checkpoint import ensure_downloaded, get_state_dict_prefix
from ml.utils.timer import Timer
from torch import Tensor, nn

WAVEGLOW_CKPT_FP16 = "https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_amp/versions/19.09.0/files/nvidia_waveglowpyt_fp16_20190427"
WAVEGLOW_CKPT_FP32 = "https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_fp32/versions/19.09.0/files/nvidia_waveglowpyt_fp32_20190427"


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a: Tensor, input_b: Tensor, n_channels: int) -> Tensor:
    n_channels_int = n_channels
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WaveGlowLoss(nn.Module):
    def __init__(self, sigma: float = 1.0) -> None:
        super().__init__()

        self.sigma = sigma

    def forward(self, model_output: tuple[Tensor, list[Tensor], list[Tensor]]) -> Tensor:
        z, log_s_list, log_det_w_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_w_total = log_det_w_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_w_total += log_det_w_list[i]

        loss = torch.sum(z * z) / (2 * self.sigma * self.sigma) - log_s_total - log_det_w_total
        return loss / (z.size(0) * z.size(1) * z.size(2))


class Invertible1x1Conv(nn.Module):
    weight_inv: Tensor

    def __init__(self, c: int) -> None:
        super().__init__()

        self.conv = nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0, bias=False)

        # Sample a random orthonormal matrix to initialize weights
        weight, _ = torch.linalg.qr(torch.randn(c, c), "reduced")

        # Ensure determinant is 1.0 not -1.0
        if torch.det(weight) < 0:
            weight[:, 0] = -1 * weight[:, 0]
        weight = weight.view(c, c, 1)
        self.conv.weight.data = weight

        self.register_buffer("weight_inv", torch.zeros_like(weight), persistent=False)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, _, n_of_groups = z.size()

        weight = self.conv.weight.squeeze()

        # Forward computation.
        log_det_w = batch_size * n_of_groups * torch.logdet(weight)
        z = self.conv(z)
        return z, log_det_w

    def infer(self, z: Tensor) -> Tensor:
        self._invert()
        return F.conv1d(z, self.weight_inv, bias=None, stride=1, padding=0)

    def _invert(self) -> None:
        weight = self.conv.weight.squeeze()
        self.weight_inv.copy_(weight.float().inverse().unsqueeze(-1).to(self.weight_inv))


@dataclass
class WaveNetConfig:
    n_layers: int = conf_field(8, help="Number of layers")
    kernel_size: int = conf_field(3, help="Kernel size")
    n_channels: int = conf_field(512, help="Number of channels")


class WaveNet(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        n_mel_channels: int,
        config: WaveNetConfig,
        lora_rank: int | None = None,
    ) -> None:
        super().__init__()

        assert config.kernel_size % 2 == 1
        assert config.n_channels % 2 == 0
        self.n_layers = config.n_layers
        self.n_channels = config.n_channels
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.cond_layers = nn.ModuleList()

        start = nn.Conv1d(n_in_channels, config.n_channels, 1)
        start = nn.utils.weight_norm(start, name="weight")
        self.start = maybe_lora(start, lora_rank)

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = nn.Conv1d(config.n_channels, 2 * n_in_channels, 1)
        end.weight.data.zero_()
        if end.bias is not None:
            end.bias.data.zero_()
        self.end = maybe_lora(end, lora_rank)

        for i in range(config.n_layers):
            dilation = 2**i
            padding = int((config.kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                config.n_channels, 2 * config.n_channels, config.kernel_size, dilation=dilation, padding=padding
            )
            in_layer = nn.utils.weight_norm(in_layer, name="weight")
            in_layer = maybe_lora(in_layer, lora_rank)
            self.in_layers.append(in_layer)

            cond_layer = nn.Conv1d(n_mel_channels, 2 * config.n_channels, 1)
            cond_layer = nn.utils.weight_norm(cond_layer, name="weight")
            cond_layer = maybe_lora(cond_layer, lora_rank)
            self.cond_layers.append(cond_layer)

            # last one is not necessary
            if i < config.n_layers - 1:
                res_skip_channels = 2 * config.n_channels
            else:
                res_skip_channels = config.n_channels
            res_skip_layer = nn.Conv1d(config.n_channels, res_skip_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name="weight")
            res_skip_layer = maybe_lora(res_skip_layer, lora_rank)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio: Tensor, spect: Tensor) -> Tensor:
        audio = self.start(audio)

        output = 0
        layers = zip(self.in_layers, self.cond_layers, self.res_skip_layers)
        for i, (in_layer, cond_layer, res_skip_layer) in enumerate(layers):
            acts = fused_add_tanh_sigmoid_multiply(in_layer(audio), cond_layer(spect), self.n_channels)

            res_skip_acts = res_skip_layer(acts)
            if i < self.n_layers - 1:
                audio = res_skip_acts[:, : self.n_channels, :] + audio
                skip_acts = res_skip_acts[:, self.n_channels :, :]
            else:
                skip_acts = res_skip_acts

            output += skip_acts
        return self.end(output)


@dataclass
class WaveGlowConfig:
    n_mel_channels: int = conf_field(80, help="Number of mel channels")
    n_flows: int = conf_field(12, help="Number of flows")
    n_group: int = conf_field(8, help="Number of groups in a flow")
    n_early_every: int = conf_field(4, help="Number of layers between early layers")
    n_early_size: int = conf_field(2, help="Number of channels in early layers")
    sampling_rate: int = conf_field(22050, help="Sampling rate of model.")
    wavenet: WaveNetConfig = conf_field(WaveNetConfig(), help="WaveNet configuration")
    lora_rank: int | None = conf_field(None, help="LoRA rank")


class WaveGlow(nn.Module):
    def __init__(self, config: WaveGlowConfig) -> None:
        super().__init__()

        self.sampling_rate = config.sampling_rate
        self.upsample = nn.ConvTranspose1d(config.n_mel_channels, config.n_mel_channels, 1024, stride=256)
        assert config.n_group % 2 == 0
        self.n_flows = config.n_flows
        self.n_group = config.n_group
        self.n_early_every = config.n_early_every
        self.n_early_size = config.n_early_size
        self.WN = cast(list[WaveNet], nn.ModuleList())
        self.convinv = nn.ModuleList()

        n_half = config.n_group // 2

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = config.n_group
        for k in range(config.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.WN.append(WaveNet(n_half, config.n_mel_channels * config.n_group, config.wavenet, config.lora_rank))
        self.n_remaining_channels = n_remaining_channels

    def forward(self, forward_input: tuple[Tensor, Tensor]) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        spect, audio = forward_input

        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)
        assert spect.size(2) >= audio.size(1)
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, : audio.size(1)]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
        spect = spect.permute(0, 2, 1)

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        output_audio = []
        log_s_list = []
        log_det_w_list = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, : self.n_early_size, :])
                audio = audio[:, self.n_early_size :, :]

            audio, log_det_w = self.convinv[k](audio)
            log_det_w_list.append(log_det_w)

            n_half = int(audio.size(1) // 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k](audio_0, spect)
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_w_list

    def infer(self, spect: Tensor, sigma: float = 1.0) -> Tensor:
        spect = self.upsample(spect)
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
        spect = spect.permute(0, 2, 1)

        audio = spect.new_empty(spect.size(0), self.n_remaining_channels, spect.size(2)).normal_(std=sigma)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k](audio_0, spect)
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.convinv[k].infer(audio)

            if k % self.n_early_every == 0 and k > 0:
                z = torch.randn(spect.size(0), self.n_early_size, spect.size(2), device=spect.device).to(spect.dtype)
                audio = torch.cat((sigma * z, audio), 1)

        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1).data
        return audio

    def remove_weight_norm(self) -> None:
        """Removes weight normalization module from all of the WaveGlow modules."""

        def remove(conv_list: nn.ModuleList) -> nn.ModuleList:
            new_conv_list = nn.ModuleList()
            for old_conv in conv_list:
                old_conv = nn.utils.remove_weight_norm(old_conv)
                new_conv_list.append(old_conv)
            return new_conv_list

        for wave_net in self.WN:
            wave_net.start = nn.utils.remove_weight_norm(wave_net.start)
            wave_net.in_layers = remove(wave_net.in_layers)
            wave_net.cond_layers = remove(wave_net.cond_layers)
            wave_net.res_skip_layers = remove(wave_net.res_skip_layers)


def pretrained_waveglow(
    *,
    fp16: bool = True,
    pretrained: bool = True,
    lora_rank: int | None = None,
) -> WaveGlow:
    """Loads the pretrained WaveGlow model.

    Reference:
        https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/waveglow/entrypoints.py

    Args:
        fp16: When True, returns a model with half precision float16 weights
        pretrained: When True, returns a model pre-trained on LJ Speech dataset
        lora_rank: The LoRA rank to use, if LoRA is desired.

    Returns:
        The WaveGlow model
    """
    config = WaveGlowConfig(lora_rank=lora_rank)
    model = WaveGlow(config)

    if pretrained:
        weights_name = f"weights_fp{16 if fp16 else 32}.pth"
        with Timer("downloading checkpoint"):
            fpath = ensure_downloaded(WAVEGLOW_CKPT_FP16 if fp16 else WAVEGLOW_CKPT_FP32, "waveglow", weights_name)
        ckpt = torch.load(fpath, map_location="cpu")
        model.load_state_dict(get_state_dict_prefix(ckpt["state_dict"], "module."))

    return model
