"""Runs tests for HiFi-GAN model.

This test is to make sure that running the HiFi-GAN model in streaming mode
matches running it in batch mode.
"""

import pytest
import torch
from torch import Tensor

from pretrained.vocoder.hifigan import HiFiGAN, partial_conv


def test_streaming_matches_batch() -> None:
    bsz, tsz = 2, 15
    model = HiFiGAN()
    model.double()
    x = torch.randn(bsz, model.model_in_dim, tsz, dtype=torch.double)
    batch_y = model.forward(x)
    infer_y = model.infer(x)
    assert torch.allclose(batch_y, infer_y)
