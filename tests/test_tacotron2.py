"""Tests that Tacotron training matches inference."""

import torch

from pretrained.tacotron2 import pretrained_tacotron2


def test_training_matches_inference() -> None:
    model = pretrained_tacotron2(pretrained=False, prenet_dropout=False)
    model.eval()

    memory = torch.randn(1, 10, 512)
    memory_lengths = torch.tensor([10])
    mels_infer, _, _, _ = model.decoder.infer(memory, memory_lengths)
    mels_train, _, _, _ = model.decoder.forward(memory, mels_infer, memory_lengths)
    assert torch.allclose(mels_infer, mels_train)
