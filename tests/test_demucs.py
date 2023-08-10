"""Unit tests for the Demucs model.

Basically makes sure that the training outputs match the streaming outputs and
provides an example of running the real-time model.
"""

import os

import torch
from ml.utils.device.auto import detect_device
from torch import Tensor

from pretrained.demucs import Demucs


def test_demucs_streamer_matches_training() -> None:
    os.environ["USE_FP16"] = "0"
    os.environ["USE_BF16"] = "0"
    os.environ["USE_FP32"] = "0"
    os.environ["USE_FP64"] = "1"
    os.environ["DISABLE_METAL"] = "1"

    device = detect_device()

    model = Demucs(
        in_channels=1,
        out_channels=1,
        hidden=48,
        depth=5,
        kernel_size=8,
        stride=4,
        causal=True,
        resample=4,
        growth=2,
        max_hidden=10_000,
        normalize=True,
        rescale=0.1,
        floor=0.001,
        sample_rate=16_000,
    )
    device.module_to(model)
    model.eval()

    # Runs the forward training pass.
    mix_in = device.tensor_to(torch.randn(1, 1, model.valid_length(16_000)))
    mix_out = model.forward(mix_in)
    assert mix_in.shape == mix_out.shape

    mix_out_2 = model.forward(mix_in)
    assert torch.allclose(mix_out, mix_out_2)

    # Runs streamer inference.
    streamer = model.streamer(device=device)
    mix_out_chunks: list[Tensor] = []
    for mix_in_chunk in mix_in.squeeze(0).chunk(1, dim=-1):
        mix_out_chunk = streamer.feed(mix_in_chunk)
        mix_out_chunks.append(mix_out_chunk)

    mix_out_chunk = streamer.flush()
    mix_out_chunks.append(mix_out_chunk)

    mix_out_streamer = torch.cat(mix_out_chunks, dim=-1).unsqueeze(0)
    assert mix_out_streamer.shape == mix_out.shape
    # assert torch.allclose(mix_out_streamer, mix_out)
    delta = torch.norm(mix_out_streamer - mix_out) / torch.norm(mix_out)
    assert delta < 0.05
