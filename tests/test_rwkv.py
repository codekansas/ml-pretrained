"""Tests the RWKV model.

This test checks that the RWKV model training works as expected - for example,
in the original code, there wasn't a pure-Python implementation of the `wkv`
operation. The one implemented in this package needs to match the CUDA kernel.
Rather than evaluating the CUDA kernel on some samples and checking that the
Python version matches, this test simply checks that the iterative version
matches the batched version.
"""

import pytest
import torch
from torch import Tensor

from pretrained.rwkv import run_wkv


def test_wkv() -> None:
    bsz, tsz, chans = 2, 7, 16

    # Gets some dummy tensors.
    w, u = torch.rand(chans), torch.rand(chans)
    k, v = torch.randn(bsz, tsz, chans), torch.randn(bsz, tsz, chans)
    num, den = torch.randn(bsz, 1, chans), torch.randn(bsz, 1, chans)

    # Runs in full mode.
    out_full, _, _ = run_wkv(w, u, k, v, num, den)

    # Runs in iterative mode.
    out_parts: list[Tensor] = []
    for t in range(tsz):
        out_part, num, den = run_wkv(w, u, k[:, t : t + 1], v[:, t : t + 1], num, den)
        out_parts.append(out_part)
    out_partial = torch.cat(out_parts, dim=1)

    breakpoint()
    assert torch.allclose(out_full, out_partial)


@pytest.mark.has_gpu()
def test_kernel_matches_ref() -> None:
    bsz, tsz, chans = 2, 7, 16
    device = torch.device("cuda")
    mask = get_mask(tsz, device=device)

    # Gets some dummy tensors.
    w, u = torch.rand(chans, device=device), torch.rand(chans, device=device)
    k, v = torch.randn(bsz, tsz, chans, device=device), torch.randn(bsz, tsz, chans, device=device)
    last_num, last_den = torch.randn(bsz, 1, chans, device=device), torch.randn(bsz, 1, chans, device=device)

    ref_out = run_wkv_train(w, u, k, v, last_num, last_den, mask, use_cuda_if_available=False)
    cuda_out = run_wkv_train(w, u, k, v, last_num, last_den, mask, use_cuda_if_available=True)

    assert torch.allclose(ref_out, cuda_out)


if __name__ == "__main__":
    # python -m tests.test_rwkv
    test_kernel_matches_ref()
