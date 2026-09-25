# SPDX-License-Identifier: Apache-2.0
"""Packed GDN state for VLLM_DAMP_STATE=mixed.

The page is one uint8 tensor. Decode reads and writes it in packed channel
order. Prefill still hands the chunk kernel a dense fp16 row, packed and
unpacked here with vector loads. Index 0 stays the null block.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.mamba.damp_pack import indices_from_mask, load_mask

_V = 128
_HEADS = 48

_hi_idx: torch.Tensor | None = None
_lo_idx: torch.Tensor | None = None
_n_hi = 0
_n_lo = 0
_ctx: dict | None = None


def damp_enabled() -> bool:
    return os.environ.get("VLLM_DAMP_STATE", "off") == "mixed"


def _ensure(device: torch.device, heads: int) -> None:
    global _hi_idx, _lo_idx, _n_hi, _n_lo
    if _hi_idx is not None:
        return
    path = os.environ.get("DAMP_MASK")
    if not path:
        raise RuntimeError("VLLM_DAMP_STATE=mixed requires DAMP_MASK")
    mask = load_mask(path)
    hi_idx, lo_idx, n_hi = indices_from_mask(mask)
    _n_hi = n_hi
    _n_lo = mask.shape[1] - n_hi
    _hi_idx = hi_idx.to(device=device, dtype=torch.int32)
    _lo_idx = lo_idx.to(device=device, dtype=torch.int32)
    if heads != mask.shape[0]:
        raise RuntimeError(f"DAMP mask has {mask.shape[0]} heads, got {heads}")


def ensure_warmup(device: torch.device, heads: int) -> None:
    if damp_enabled():
        _ensure(device, heads)


def kernel_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float16 if dtype == torch.uint8 else dtype


def _views(page: torch.Tensor):
    _ensure(page.device, _HEADS)
    blocks = page.shape[0]
    hi_b = _HEADS * _V * _n_hi * 2
    lo_b = _HEADS * _V * _n_lo
    if _n_hi:
        hi = page[:, :hi_b].view(torch.float16).view(blocks, _HEADS, _V, _n_hi)
    else:
        hi = page.new_empty(blocks, _HEADS, _V, 0, dtype=torch.float16)
    lo = page[:, hi_b : hi_b + lo_b].view(torch.int8).view(blocks, _HEADS, _V, _n_lo)
    scale = page[:, hi_b + lo_b :].view(torch.float32).view(blocks, _HEADS, _V // 32)
    return hi, lo, scale


def _rest(rows: torch.Tensor):
    return rows.shape[1:]


def _flat(indices: torch.Tensor) -> torch.Tensor:
    # Rank 1 may be a column of the align-mode block table (stride = row width).
    # The kernels take that stride. Do not copy: a new buffer aborts CUDA graph capture.
    if indices.ndim == 1:
        return indices
    if not indices.is_contiguous():
        raise RuntimeError("DAMP state indices above rank 1 must be contiguous")
    return indices.reshape(-1)


def _index_stride(indices: torch.Tensor) -> int:
    if indices.ndim == 1 and indices.numel():
        return int(indices.stride(0))
    return 1


@triton.jit
def _read_kernel(
    hi, lo, scale, dense, index, index_stride, hi_idx, lo_idx,
    stride_hi, stride_lo, stride_scale,
    heads: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
    N_HI: tl.constexpr, N_LO: tl.constexpr,
):
    i = tl.program_id(0)
    h = tl.program_id(1)
    block = tl.load(index + i * index_stride).to(tl.int64)
    rows = tl.program_id(2) * 32 + tl.arange(0, 32)
    pos = tl.arange(0, 128)
    dest = dense + ((i * heads + h) * V) * K
    if block <= 0:
        tl.store(dest + rows[:, None] * K + pos[None, :], tl.zeros([32, 128], tl.float16),
                 mask=(rows[:, None] < V) & (pos[None, :] < K))
        return
    sc = tl.load(scale + block * stride_scale + h * 4 + tl.program_id(2)).to(tl.float32)
    hi_col = tl.load(hi_idx + h * N_HI + pos, mask=pos < N_HI, other=0)
    lo_col = tl.load(lo_idx + h * N_LO + tl.maximum(pos - N_HI, 0), mask=(pos >= N_HI) & (pos < K), other=0)
    col = tl.where(pos < N_HI, hi_col, lo_col)
    hi_vals = tl.load(
        hi + block * stride_hi + (h * V + rows[:, None]) * N_HI + pos[None, :],
        mask=(rows[:, None] < V) & (pos[None, :] < N_HI), other=0,
    ).to(tl.float32)
    lo_vals = tl.load(
        lo + block * stride_lo + (h * V + rows[:, None]) * N_LO + tl.maximum(pos - N_HI, 0)[None, :],
        mask=(rows[:, None] < V) & (pos[None, :] >= N_HI) & (pos[None, :] < K), other=0,
    ).to(tl.float32) * sc
    vals = tl.where(pos[None, :] < N_HI, hi_vals, lo_vals).to(tl.float16)
    tl.store(
        dest + rows[:, None] * K + col[None, :],
        vals,
        mask=(rows[:, None] < V) & (pos[None, :] < K),
    )


@triton.jit
def _write_kernel(
    hi, lo, scale, dense, index, index_stride, hi_idx, lo_idx,
    stride_hi, stride_lo, stride_scale, dense_stride0,
    heads: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
    N_HI: tl.constexpr, N_LO: tl.constexpr,
):
    i = tl.program_id(0)
    h = tl.program_id(1)
    block = tl.load(index + i * index_stride).to(tl.int64)
    if block <= 0:
        return
    offs = tl.arange(0, 128)
    base = dense + i * dense_stride0 + h * V * K
    if N_LO > 0:
        lo_col = tl.load(lo_idx + h * N_LO + offs, mask=offs < N_LO, other=0)
        for tile in range(0, 4):
            rows = tile * 32 + tl.arange(0, 32)
            vals = tl.load(
                base + rows[:, None] * K + lo_col[None, :],
                mask=(rows[:, None] < V) & (offs[None, :] < N_LO), other=0,
            ).to(tl.float32)
            amax = tl.max(tl.abs(vals))
            sc = tl.maximum(amax, 1e-8) / 127.0
            tl.store(scale + block * stride_scale + h * 4 + tile, sc)
            qv = vals / sc
            q = tl.where(qv >= 0, tl.floor(qv + 0.5), tl.ceil(qv - 0.5))
            q = tl.minimum(tl.maximum(q, -127.0), 127.0)
            tl.store(
                lo + block * stride_lo + (h * V + rows[:, None]) * N_LO + offs[None, :],
                q.to(tl.int8),
                mask=(rows[:, None] < V) & (offs[None, :] < N_LO),
            )
    if N_HI > 0:
        hi_col = tl.load(hi_idx + h * N_HI + offs, mask=offs < N_HI, other=0)
        for v0 in range(0, V, 32):
            rows = v0 + tl.arange(0, 32)
            vals = tl.load(
                base + rows[:, None] * K + hi_col[None, :],
                mask=(rows[:, None] < V) & (offs[None, :] < N_HI), other=0,
            )
            tl.store(
                hi + block * stride_hi + (h * V + rows[:, None]) * N_HI + offs[None, :],
                vals,
                mask=(rows[:, None] < V) & (offs[None, :] < N_HI),
            )


def _launch_read(hi, lo, scale, dense, flat):
    s = flat.numel()
    _read_kernel[(s, _HEADS, triton.cdiv(_V, 32))](
        hi, lo, scale, dense, flat, _index_stride(flat), _hi_idx, _lo_idx,
        hi.stride(0), lo.stride(0), scale.stride(0),
        heads=_HEADS, V=_V, K=_n_hi + _n_lo, N_HI=_n_hi, N_LO=_n_lo,
    )


def _launch_write(hi, lo, scale, dense, flat):
    s = flat.numel()
    _write_kernel[(s, _HEADS)](
        hi, lo, scale, dense, flat, _index_stride(flat), _hi_idx, _lo_idx,
        hi.stride(0), lo.stride(0), scale.stride(0), dense.stride(0),
        heads=_HEADS, V=_V, K=_n_hi + _n_lo, N_HI=_n_hi, N_LO=_n_lo,
    )


def damp_read_rows(page: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if page.dtype != torch.uint8:
        return page[indices]
    hi, lo, scale = _views(page)
    flat = _flat(indices)
    dense = torch.empty((flat.numel(), _HEADS, _V, _n_hi + _n_lo), dtype=torch.float16, device=page.device)
    _launch_read(hi, lo, scale, dense, flat)
    return dense.view(*indices.shape, _HEADS, _V, _n_hi + _n_lo)


def damp_write_rows(page: torch.Tensor, indices: torch.Tensor, dense: torch.Tensor) -> None:
    if page.dtype != torch.uint8:
        page[indices] = dense.to(page.dtype)
        return
    hi, lo, scale = _views(page)
    flat = _flat(indices)
    _launch_write(hi, lo, scale, dense.reshape(flat.numel(), _HEADS, _V, -1), flat)


def damp_prepare(page: torch.Tensor, indices: torch.Tensor):
    """Dense fp16 rows for a backend that cannot read the page. Decode does not use this."""
    global _ctx
    if page.dtype != torch.uint8:
        return page, indices
    flat = _flat(indices)
    rows = damp_read_rows(page, flat)
    dense = torch.zeros((flat.numel() + 1, *_rest(rows)), dtype=rows.dtype, device=rows.device)
    dense[1:] = rows
    slots = torch.arange(1, flat.numel() + 1, device=page.device, dtype=torch.int32)
    slots = torch.where(flat > 0, slots, torch.zeros_like(slots))
    _ctx = {"page": page, "flat": flat, "dense": dense}
    return dense, slots.view(indices.shape)


def damp_commit() -> None:
    global _ctx
    if _ctx is None:
        return
    ctx = _ctx
    _ctx = None
    damp_write_rows(ctx["page"], ctx["flat"], ctx["dense"][1:])
