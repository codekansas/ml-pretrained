"""Unit tests for the causal HuBERT model.

This model simply checks that the streaming version of the model matches the
batched version of the model.
"""

import torch

from pretrained.causal_hubert import CausalHubert, CausalHubertState, ConvExtractor


def test_causal_hubert_streamer_matches_training() -> None:
    bsz, tsz = 2, 16_000

    model = CausalHubert(
        hidden_size=32,
        num_heads=2,
        dim_feedforward=128,
        num_layers=2,
        num_hubert_tokens=10,
        extractor=ConvExtractor(hidden_size=32),
        local_attn=4,
    )
    model.double()

    # Runs the forward pass in batched mode.
    waveform = torch.randn(bsz, tsz, dtype=torch.double)
    tokens, _ = model.forward(waveform)

    # Runs the forward pass in streaming mode.
    state: CausalHubertState | None = None
    token_chunks: list[torch.Tensor] = []
    for waveform_chunk in waveform.chunk(10, dim=1):
        token_chunk, state = model.forward(waveform_chunk, state)
        token_chunks.append(token_chunk)

    # Checks that the resulting tokens are equal.
    tokens_streaming = torch.cat(token_chunks, dim=1)
    assert torch.allclose(tokens, tokens_streaming)
