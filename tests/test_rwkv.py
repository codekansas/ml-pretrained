"""Tests the RWKV model.

This test checks that the RWKV model training works as expected - for example,
in the original code, there wasn't a pure-Python implementation of the `wkv`
operation. The one implemented in this package needs to match the CUDA kernel.
Rather than evaluating the CUDA kernel on some samples and checking that the
Python version matches, this test simply checks that the iterative version
matches the batched version.
"""

import torch
from torch import Tensor

from pretrained.rwkv import get_mask, run_wkv


def test_wkv() -> None:
    bsz, tsz, chans = 2, 7, 16
    mask = get_mask(tsz)

    # Gets some dummy tensors.
    # w, u = torch.rand(chans), torch.rand(chans)
    w, u = torch.arange(chans).flip(0).double(), torch.arange(chans).flip(0).double()
    u = torch.ones_like(u)
    # k, v = torch.randn(bsz, tsz, chans), torch.randn(bsz, tsz, chans)
    k = torch.arange(tsz * chans).double().view(1, tsz, chans).repeat(bsz, 1, 1) / (tsz * chans)
    v = torch.arange(tsz * chans).double().view(1, tsz, chans).repeat(bsz, 1, 1) / (tsz * chans)
    # last_num, last_den = torch.randn(bsz, 1, chans), torch.randn(bsz, 1, chans)
    last_num = torch.arange(chans).double().view(1, 1, -1).repeat(bsz, 1, 1)
    last_den = torch.arange(chans).double().view(1, 1, -1).repeat(bsz, 1, 1)

    out_full, out_num, out_den = run_wkv(tsz, w, u, k, v, last_num, last_den, mask)
    out_parts: list[Tensor] = []
    out_nums: list[Tensor] = []
    out_dens: list[Tensor] = []
    last_num_t, last_den_t = last_num, last_den
    for t in range(tsz):
        out_part, last_num_t, last_den_t = run_wkv(1, w, u, k[:, t : t + 1], v[:, t : t + 1], last_num_t, last_den_t)
        out_parts.append(out_part)
        out_nums.append(last_num_t)
        out_dens.append(last_den_t)

    out_partial = torch.cat(out_parts, dim=1)
    out_num_part = torch.cat(out_nums, dim=1)
    out_den_part = torch.cat(out_dens, dim=1)

    out_full - out_partial
    out_num - out_num_part
    out_den - out_den_part

    assert torch.allclose(out_full, out_partial)
    assert torch.allclose(out_num, out_num_part)
    assert torch.allclose(out_den, out_den_part)
