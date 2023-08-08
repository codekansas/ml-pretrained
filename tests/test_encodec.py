"""Runs some simple tests on the Encodec model."""

import torch

from pretrained.encodec import pretrained_encodec


def test_pretrained_encodec() -> None:
    model = pretrained_encodec("24khz", load_weights=False)
    encoder, decoder = model.get_encoder(), model.get_decoder()

    waveform = torch.randn(1, 1, 24000)
    tokens = encoder(waveform)
    reconstructed = decoder(tokens)

    assert tokens.shape[:2] == (1, 32)
    assert reconstructed.shape == waveform.shape

    reconstructed_train, vq_losses = model(waveform)
    assert reconstructed_train.shape == waveform.shape
    assert vq_losses.shape == (32, 1)
