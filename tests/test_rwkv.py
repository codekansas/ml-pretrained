"""Tests the numerically stable WKV Triton kernels."""

from typing import cast

import pytest
import torch
from torch import Tensor

from pretrained.rwkv import (
    WkvFnKey,
    initial_state_log_space,
    initial_state_with_eps,
    wkv_log_space,
    wkv_log_space_forward,
    wkv_with_eps,
    wkv_with_eps_forward,
)


def _get_dummy_tensors(bsz: int, tsz: int, chans: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, ...]:
    w = torch.exp(torch.rand(chans, dtype=dtype, device=device))
    u = torch.rand(chans, dtype=dtype, device=device)
    k = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    v = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    return w, u, k, v


def _copy_with_grad(*t: Tensor) -> tuple[Tensor, ...]:
    return tuple(t_i.detach().clone().requires_grad_(True) for t_i in t)


def _get_grads(*t: Tensor) -> tuple[Tensor | None, ...]:
    return tuple(cast(Tensor, t_i.grad) for t_i in t)


@pytest.mark.parametrize("key", ["eps", "log"])
def test_eps_wkv(key: WkvFnKey) -> None:
    bsz, tsz, chans = 2, 7, 16
    device, dtype = torch.device("cpu"), torch.float32

    match key:
        case "eps":
            wkv_fn = wkv_with_eps
            initial_state_fn = initial_state_with_eps
        case "log":
            wkv_fn = wkv_log_space
            initial_state_fn = initial_state_log_space
        case _:
            raise ValueError(f"Invalid key: {key}")

    w, u, k, v = _get_dummy_tensors(bsz, tsz, chans, device, dtype)
    state = initial_state_fn(chans).repeat_interleave(bsz, dim=0).to(device, dtype)

    # Runs in full mode.
    out_full, _ = wkv_fn(w, u, k, v, state)

    # Runs in iterative mode.
    out_parts: list[Tensor] = []
    for t in range(tsz):
        out_part, state = wkv_fn(w, u, k[:, t : t + 1], v[:, t : t + 1], state)
        out_parts.append(out_part)
    out_partial = torch.cat(out_parts, dim=1)

    assert torch.allclose(out_full, out_partial, atol=1e-5)


@pytest.mark.parametrize("mode", ["state", "wkv", "both"])
@pytest.mark.parametrize("key", ["eps", "log"])
def test_gradients_wkv(mode: str, key: WkvFnKey) -> None:
    bsz, tsz, chans = 2, 7, 16
    device, dtype = torch.device("cpu"), torch.float32

    match key:
        case "eps":
            wkv_fn = wkv_with_eps
            wkv_forward_fn = wkv_with_eps_forward
            initial_state_fn = initial_state_with_eps
        case "log":
            wkv_fn = wkv_log_space
            wkv_forward_fn = wkv_log_space_forward
            initial_state_fn = initial_state_log_space
        case _:
            raise ValueError(f"Invalid key: {key}")

    w, u, k, v = _get_dummy_tensors(bsz, tsz, chans, device, dtype)
    state = initial_state_fn(chans).repeat_interleave(bsz, dim=0).to(device, dtype)

    # Uses autograd to compute the gradients.
    wt, ut, kt, vt, statet = _copy_with_grad(w, u, k, v, state)
    wkv_ref, state_out_ref = wkv_forward_fn(wt, ut, kt, vt, statet)
    state_out_ref = state_out_ref[:, :, -1:]
    wkv_grad = torch.zeros_like(wkv_ref) if mode == "state" else torch.rand_like(wkv_ref)
    state_out_grad = torch.zeros_like(state_out_ref) if mode == "wkv" else torch.rand_like(state_out_ref)
    torch.autograd.backward((wkv_ref, state_out_ref), (wkv_grad, state_out_grad))
    wgr, ugr, kgr, vgr, stategr = _get_grads(wt, ut, kt, vt, statet)

    # Uses the manual gradient computation to compute the gradients.
    wt, ut, kt, vt, statet = _copy_with_grad(w, u, k, v, state)
    wkv_man, state_out_man = wkv_fn(wt, ut, kt, vt, statet)
    torch.autograd.backward((wkv_man, state_out_man), (wkv_grad, state_out_grad))
    wgm, ugm, kgm, vgm, stategm = _get_grads(wt, ut, kt, vt, statet)

    for gr, gm in zip((wgr, ugr, kgr, vgr, stategr), (wgm, ugm, kgm, vgm, stategm)):
        if gr is not None and gm is not None:
            assert torch.allclose(gr, gm, atol=1e-5)


@pytest.mark.has_triton()
@pytest.mark.parametrize("tsz", [1, 4])
@pytest.mark.parametrize("key", ["eps", "log"])
def test_triton_wkv(tsz: int, key: WkvFnKey) -> None:
    from pretrained.triton.rwkv_kernel import wkv_triton_log_space, wkv_triton_with_eps

    bsz, chans = 2, 768
    device, dtype = torch.device("cuda"), torch.float32

    match key:
        case "eps":
            wkv_cpu_fn = wkv_with_eps
            wkv_triton_fn = wkv_triton_with_eps
            initial_state_fn = initial_state_with_eps
        case "log":
            wkv_cpu_fn = wkv_log_space
            wkv_triton_fn = wkv_triton_log_space
            initial_state_fn = initial_state_log_space
        case _:
            raise ValueError(f"Invalid key: {key}")

    w, u, k, v = _get_dummy_tensors(bsz, tsz, chans, device, dtype)
    state = initial_state_fn(chans).repeat_interleave(bsz, dim=0).to(device, dtype)

    wa, ua, ka, va, statea = _copy_with_grad(w, u, k, v, state)
    wkv_ref, state_out_ref = wkv_cpu_fn(wa, ua, ka, va, statea)
    wb, ub, kb, vb, stateb = _copy_with_grad(w, u, k, v, state)
    wkv, state_out = wkv_triton_fn(wb, ub, kb, vb, stateb)

    assert torch.allclose(wkv_ref, wkv, atol=1e-5)
    assert torch.allclose(state_out_ref, state_out, atol=1e-5)

    grad_wkv = torch.randn_like(wkv)
    grad_state = torch.randn_like(state_out)

    # Backpropagates the gradients.
    torch.autograd.backward((wkv, state_out), (grad_wkv, grad_state))
    torch.autograd.backward((wkv_ref, state_out_ref), (grad_wkv, grad_state))

    for a, b, name in [
        (wa.grad, wb.grad, "dw"),
        (ua.grad, ub.grad, "du"),
        (ka.grad, kb.grad, "dk"),
        (va.grad, vb.grad, "dv"),
        (statea.grad, stateb.grad, "state"),
    ]:
        assert a is not None and b is not None, f"{name} is None!"
        assert torch.allclose(a, b, atol=1e-5), f"{name} is not close!"
