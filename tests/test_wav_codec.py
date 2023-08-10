"""Runs some simple tests on the WavCodec model."""

import torch

from pretrained.wav_codec import pretrained_wav_codec


def test_pretrained_wav_codec() -> None:
    model = pretrained_wav_codec("librivox", load_weights=False)
    model.double()
    model.eval()

    # Tests on a full batch.
    waveform = torch.randn(2, 32000, dtype=torch.double)
    reconstructed, _, _ = model(waveform)
    assert reconstructed.shape == waveform.shape

    # Tests encoder-decoder.
    encoder, decoder = model.encoder(), model.decoder()
    tokens, waveform_leftover = encoder.encode(waveform)
    decoded, _ = decoder.decode(tokens)
    decoded = torch.cat([decoded, waveform_leftover], dim=1)

    assert torch.allclose(decoded, reconstructed)


if __name__ == "__main__":
    # python -m tests.test_wav_codec
    test_pretrained_wav_codec()
