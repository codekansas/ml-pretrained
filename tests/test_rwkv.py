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

from pretrained.rwkv import _wkv_triton, _wkv_vanilla


def _get_dummy_tensors(bsz: int, tsz: int, chans: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, ...]:
    w = torch.rand(chans, dtype=dtype, device=device)
    u = torch.rand(chans, dtype=dtype, device=device)
    k = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    v = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    num = torch.randn(bsz, 1, chans, dtype=dtype, device=device)
    den = torch.randn(bsz, 1, chans, dtype=dtype, device=device)
    return w, u, k, v, num, den


def _copy_with_grad(*t: Tensor) -> tuple[Tensor, ...]:
    return tuple(t_i.detach().clone().requires_grad_(True) for t_i in t)


def test_wkv() -> None:
    bsz, tsz, chans = 2, 7, 16
    w, u, k, v, num, den = _get_dummy_tensors(bsz, tsz, chans, torch.device("cpu"), torch.float32)

    # Runs in full mode.
    out_full, _, _ = _wkv_vanilla(w, u, k, v, num, den)

    # Runs in iterative mode.
    out_parts: list[Tensor] = []
    for t in range(tsz):
        out_part, num, den = _wkv_vanilla(w, u, k[:, t : t + 1], v[:, t : t + 1], num, den)
        out_parts.append(out_part)
    out_partial = torch.cat(out_parts, dim=1)

    assert torch.allclose(out_full, out_partial)


@pytest.mark.has_gpu()
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_kernel_matches_ref(dtype: torch.dtype) -> None:
    bsz, tsz, chans = 2, 7, 16
    w, u, k, v, num, den = _get_dummy_tensors(bsz, tsz, chans, torch.device("cuda"), dtype)

    # Check the forward pass against the reerence implementation.
    wr, ur, kr, vr, numr, denr = _copy_with_grad(w, u, k, v, num, den)
    ref_out, ref_num, ref_den = _wkv_vanilla(wr, ur, kr, vr, numr, denr)
    wc, uc, kc, vc, numc, denc = _copy_with_grad(w, u, k, v, num, den)
    cuda_out, cuda_num, cuda_den = _wkv_triton(wc, uc, kc, vc, numc, denc)

    for ref, cuda, name in zip((ref_out, ref_num, ref_den), (cuda_out, cuda_num, cuda_den), ("out", "num", "den")):
        assert torch.allclose(ref, cuda, atol=1e-6), f"{name} mismatch"

    # Checks the backwards passes also match.
    (ref_out.sum() + ref_num.sum() + ref_den.sum()).backward()
    (cuda_out.sum() + cuda_num.sum() + cuda_den.sum()).backward()

    # for ref, cuda in zip((wr, ur, kr, vr, numr, denr), (wc, uc, kc, vc, numc, denc)):
    #     assert torch.allclose(ref.grad, cuda.grad, atol=1e-6)
    for ref, cuda in zip((numr, denr), (numc, denc)):
        assert torch.allclose(ref.grad, cuda.grad, atol=1e-6)


@pytest.mark.has_gpu()
def test_triton_gradients() -> None:
    bsz, tsz, chans = 2, 7, 16
    w, u, k, v, num, den = _get_dummy_tensors(bsz, tsz, chans, torch.device("cuda"), torch.float32)

    # Runs the forward pass and backpropagates.
    cuda_out, cuda_num, cuda_den = _wkv_triton(w, u, k, v, num, den)
    (cuda_out.sum() + cuda_num.sum() + cuda_den.sum()).backward()

    # Checks the gradients against finite difference approximation.
    assert torch.autograd.gradcheck(
        func=_wkv_triton,
        inputs=(w, u, k, v, num, den),
        eps=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    # python -m tests.test_rwkv
    test_kernel_matches_ref(torch.float32)
