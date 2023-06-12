# ruff: noqa: ANN001, ANN201, ANN202, N803, N806
"""Defines a Triton kernel for the RWKV forward and backward passes."""

import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx


@triton.jit
def _forward_kernel(
    w_ptr,
    u_ptr,
    k_ptr,
    v_ptr,
    num_ptr,
    den_ptr,
    chans,
    tsz,
    out_ptr,
    num_out_ptr,
    den_out_ptr,
):
    # Parallelize over the batch and channel dimensions.
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    chans_tsz = chans * tsz

    # Pointers to the batch (and possibly channel) for the input tensors.
    k_ptr = k_ptr + b_idx * chans_tsz + c_idx
    v_ptr = v_ptr + b_idx * chans_tsz + c_idx
    num_ptr = num_ptr + b_idx * chans + c_idx
    den_ptr = den_ptr + b_idx * chans + c_idx
    w_ptr = w_ptr + c_idx
    u_ptr = u_ptr + c_idx

    # Pointers to the batch (and possibly channel) for the output tensors.
    out_ptr = out_ptr + b_idx * chans_tsz + c_idx
    num_out_ptr = num_out_ptr + b_idx * chans + c_idx
    den_out_ptr = den_out_ptr + b_idx * chans + c_idx

    # Loads parameters.
    num = tl.load(num_ptr).to(tl.float32)
    den = tl.load(den_ptr).to(tl.float32)
    w = -tl.exp(tl.load(w_ptr).to(tl.float32))
    ew = tl.exp(w)
    u = tl.load(u_ptr).to(tl.float32)

    for t in range(tsz):
        kt = tl.load(k_ptr + t * chans).to(tl.float32)
        vt = tl.load(v_ptr + t * chans).to(tl.float32)
        ek = tl.exp(kt)
        euk = tl.exp(u + kt)
        out = (num + euk * vt) / (den + euk)
        tl.store(out_ptr + t * chans, out)
        num = ew * num + ek * vt
        den = ew * den + ek

    tl.store(num_out_ptr, num)
    tl.store(den_out_ptr, den)


def _forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    num: Tensor,
    den: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    # New tensors to output.
    out = k.new_empty(bsz, tsz, chans)
    num_out = k.new_empty(bsz, 1, chans)
    den_out = k.new_empty(bsz, 1, chans)

    _forward_kernel[(bsz, chans)](
        w,
        u,
        k,
        v,
        num,
        den,
        chans,
        tsz,
        out,
        num_out,
        den_out,
    )

    return out, num_out, den_out


@triton.jit
def _backward_kernel(
    w_ptr,
    u_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    num_out_ptr,
    den_out_ptr,
    chans,
    tsz,
    gout_ptr,
    gnum_out_ptr,
    gden_out_ptr,
    gw_ptr,
    gu_ptr,
    gk_ptr,
    gv_ptr,
    gnum_ptr,
    gden_ptr,
):
    # Parallelize over the batch and channel dimensions.
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    chans_tsz = chans * tsz

    # Pointers to the batch (and possibly channel) for the input tensors.
    k_ptr = k_ptr + b_idx * chans_tsz + c_idx
    v_ptr = v_ptr + b_idx * chans_tsz + c_idx
    out_ptr = out_ptr + b_idx * chans_tsz + c_idx
    num_out_ptr = num_out_ptr + b_idx * chans + c_idx
    den_out_ptr = den_out_ptr + b_idx * chans + c_idx
    w_ptr = w_ptr + c_idx
    u_ptr = u_ptr + c_idx
    gout_ptr = gout_ptr + b_idx * chans_tsz + c_idx
    gnum_out_ptr = gnum_out_ptr + b_idx * chans + c_idx
    gden_out_ptr = gden_out_ptr + b_idx * chans + c_idx

    # Pointers to the batch (and possibly channel) for the output tensors.
    gw_ptr = gw_ptr + c_idx
    gu_ptr = gu_ptr + c_idx
    gk_ptr = gk_ptr + b_idx * chans_tsz + c_idx
    gv_ptr = gv_ptr + b_idx * chans_tsz + c_idx
    gnum_ptr = gnum_ptr + b_idx * chans + c_idx
    gden_ptr = gden_ptr + b_idx * chans + c_idx

    # Loads parameters.
    out = tl.load(out_ptr).to(tl.float32)
    num_out = tl.load(num_out_ptr).to(tl.float32)
    den_out = tl.load(den_out_ptr).to(tl.float32)
    gout = tl.load(gout_ptr).to(tl.float32)
    gnum_out = tl.load(gnum_out_ptr).to(tl.float32)
    gden_out = tl.load(gden_out_ptr).to(tl.float32)
    w = -tl.exp(tl.load(w_ptr).to(tl.float32))
    ew = tl.exp(w)
    u = tl.load(u_ptr).to(tl.float32)

    for t in range(tsz - 1, -1, -1):
        kt = tl.load(k_ptr + t * chans).to(tl.float32)
        vt = tl.load(v_ptr + t * chans).to(tl.float32)
        ek = tl.exp(kt)
        euk = tl.exp(u + kt)


def _backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    num_out: Tensor,
    den_out: Tensor,
    gout: Tensor,
    gnum_out: Tensor,
    gden_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    # New tensors to output.
    gw = k.new_empty(chans)
    gu = k.new_empty(chans)
    gk = k.new_empty(bsz, chans, tsz)
    gv = k.new_empty(bsz, chans, tsz)
    gnum = k.new_empty(bsz, chans)
    gden = k.new_empty(bsz, chans)

    _backward_kernel[(bsz, chans)](
        w,
        u,
        k,
        v,
        out,
        num_out,
        den_out,
        chans,
        tsz,
        gout,
        gnum_out,
        gden_out,
        gw,
        gu,
        gk,
        gv,
        gnum,
        gden,
    )

    return gw, gu, gk, gv, gnum, gden


class _WKV(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        u: Tensor,
        k: Tensor,
        v: Tensor,
        num: Tensor,
        den: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        (bsz, tsz, chans), device, dtype = k.shape, k.device, k.dtype

        # Performs tensor checks.
        for t in (k, v):
            assert t.shape == (bsz, tsz, chans)
            assert t.stride(0) == tsz * chans
            assert t.stride(1) == chans
            assert t.size(2) == 1 or t.stride(2) == 1
        for t in (num, den):
            assert t.shape == (bsz, 1, chans)
            assert t.stride(0) == chans
            assert t.stride(1) == chans
            assert t.stride(2) == 1
        for t in (w, u):
            assert t.shape == (chans,)
            assert t.stride(0) == 1
        for t in (v, num, den, w, u):
            assert t.dtype == dtype and t.device == device

        out, num_out, den_out = _forward(w, u, k, v, num, den)
        ctx.save_for_backward(w, u, k, v, out, num_out, den_out)

        return out, num_out, den_out

    @staticmethod
    def backward(ctx: FunctionCtx, gout: Tensor, gnum_out: Tensor, gden_out: Tensor) -> tuple[Tensor, ...]:
        w, u, k, v, out, num_out, den_out = ctx.saved_tensors
        gw, gu, gk, gv, gn, gd = _backward(w, u, k, v, out, num_out, den_out, gout, gnum_out, gden_out)
        return gw, gu, gk, gv, gn, gd


def triton_wkv(w: Tensor, u: Tensor, k: Tensor, v: Tensor, num: Tensor, den: Tensor) -> Tensor:
    return _WKV.apply(w, u, k, v, num, den)
