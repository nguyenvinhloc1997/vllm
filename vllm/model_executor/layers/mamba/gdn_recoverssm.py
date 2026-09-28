# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN RecoverSSM commit: replay accepted tokens from one checkpoint.

Plain (fp16/bf16/fp32) and DAMP packed pages share the record and the
replay; only the state load and store differ. Load, decay, rank-1 update and
store mirror the verify kernels statement for statement, so the committed
state equals the baseline's per-draft slot bit for bit.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.layers.mamba.gdn_recoverssm_ops import (
    _damp_quantize,
    _gdn_rank1,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
    _compact_conv_state_kernel,
    _prepare_commit_plan_kernel,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.jit
def _layer_ptr(ref, addrs, strides, layer, idx):
    base = tl.load(addrs + layer).to(tl.pointer_type(ref.dtype.element_ty))
    return base + idx * tl.load(strides + layer)


@triton.jit
def _store_plain(p, b_h, o_v, o_k, mask_h, K: tl.constexpr):
    tl.store(
        p + o_v[:, None] * K + o_k[None, :], b_h.to(p.dtype.element_ty), mask=mask_h
    )


@triton.jit
def _store_damp(
    p_hi,
    p_lo,
    p_sc,
    b_hi,
    b_lo,
    o_v,
    mask_v,
    hi_lane,
    lo_lane,
    N_HI: tl.constexpr,
    N_LO: tl.constexpr,
):
    if N_HI > 0:
        tl.store(
            p_hi + o_v[:, None] * N_HI + hi_lane[None, :],
            b_hi.to(tl.float16),
            mask=mask_v[:, None] & (hi_lane[None, :] < N_HI),
        )
    if N_LO > 0:
        codes, sc = _damp_quantize(b_lo)
        tl.store(
            p_lo + o_v[:, None] * N_LO + lo_lane[None, :],
            codes.to(tl.int8),
            mask=mask_v[:, None] & (lo_lane[None, :] < N_LO),
        )
        tl.store(p_sc, sc)


@triton.jit
def _gdn_commit_kernel(
    st_ref,  # plain: (blocks, HV, V, K); DAMP: fp16 hi view
    st_addrs,
    st_strides,
    lo_ref,  # DAMP only: int8 lo view
    lo_addrs,
    lo_strides,
    sc_ref,  # DAMP only: fp32 scale view
    sc_addrs,
    sc_strides,
    rc_ref,
    rc_addrs,
    rc_strides,
    rk_ref,
    rk_addrs,
    rk_strides,
    rd_ref,
    rd_addrs,
    rd_strides,
    state_indices_ptr,
    commit_lens_ptr,
    final_idx_ptr,
    boundary_idx_ptr,
    boundary_len_ptr,
    null_block_id,
    stride_state_indices,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BV: tl.constexpr,
    N_HI: tl.constexpr,
    N_LO: tl.constexpr,
    N_TILES: tl.constexpr,
    DAMP: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    i_v, i_b, i_lh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_l, i_hv = i_lh // HV, i_lh % HV
    src = tl.load(state_indices_ptr + i_b * stride_state_indices).to(tl.int64)
    commit_len = tl.load(commit_lens_ptr + i_b)
    dst = tl.load(final_idx_ptr + i_b).to(tl.int64)
    if (src <= null_block_id) | (commit_len == 0) | (dst <= null_block_id):
        return
    bnd = tl.load(boundary_idx_ptr + i_b).to(tl.int64)
    bnd_len = tl.load(boundary_len_ptr + i_b)

    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    mask_v = o_v < V
    mask_h = mask_v[:, None] & (o_k[None, :] < K)
    hi_lane = tl.arange(0, 32)
    lo_lane = tl.arange(0, 128)

    p_rc = _layer_ptr(rc_ref, rc_addrs, rc_strides, i_l, src) + i_hv * S * V
    p_rk = _layer_ptr(rk_ref, rk_addrs, rk_strides, i_l, src) + i_hv * S * K
    p_rd = _layer_ptr(rd_ref, rd_addrs, rd_strides, i_l, src) + i_hv * S

    # State load: same form as the verify kernels' initial-state load.
    if DAMP:
        head_hi = i_hv * V * N_HI
        head_lo = i_hv * V * N_LO
        head_sc = i_hv * N_TILES + i_v
        p_sc = _layer_ptr(sc_ref, sc_addrs, sc_strides, i_l, src) + head_sc
        b_scale = tl.load(p_sc).to(tl.float32)
        b_hi = tl.zeros([BV, 32], dtype=tl.float32)
        b_lo = tl.zeros([BV, 128], dtype=tl.float32)
        if N_HI > 0:
            p_hi = _layer_ptr(st_ref, st_addrs, st_strides, i_l, src) + head_hi
            b_hi = tl.load(
                p_hi + o_v[:, None] * N_HI + hi_lane[None, :],
                mask=mask_v[:, None] & (hi_lane[None, :] < N_HI),
                other=0,
            ).to(tl.float32)
        if N_LO > 0:
            p_lo = _layer_ptr(lo_ref, lo_addrs, lo_strides, i_l, src) + head_lo
            b_lo = (
                tl.load(
                    p_lo + o_v[:, None] * N_LO + lo_lane[None, :],
                    mask=mask_v[:, None] & (lo_lane[None, :] < N_LO),
                    other=0,
                ).to(tl.float32)
                * b_scale
            )
    else:
        head = i_hv * V * K
        p_st = _layer_ptr(st_ref, st_addrs, st_strides, i_l, src) + head
        b_h = tl.zeros([BV, K], dtype=tl.float32)
        b_h += tl.load(p_st + o_v[:, None] * K + o_k[None, :], mask=mask_h, other=0).to(
            tl.float32
        )

    for t in range(0, commit_len):
        b_d = tl.load(p_rd + t)
        b_c = tl.load(p_rc + t * V + o_v, mask=mask_v, other=0)
        if DAMP:
            k_hi = tl.load(p_rk + t * K + hi_lane, mask=hi_lane < N_HI, other=0)
            k_lo = tl.load(p_rk + t * K + N_HI + lo_lane, mask=lo_lane < N_LO, other=0)
            b_hi *= b_d
            b_lo *= b_d
            b_hi = _gdn_rank1(b_hi, b_c, k_hi)
            b_lo = _gdn_rank1(b_lo, b_c, k_lo)
        else:
            b_k = tl.load(p_rk + t * K + o_k)
            b_h *= b_d
            b_h = _gdn_rank1(b_h, b_c, b_k)
        if ALIGN_MODE:  # noqa: SIM102 (constexpr branch; `and` breaks Triton)
            if (t + 1 == bnd_len) & (bnd > null_block_id):
                if DAMP:
                    _store_damp(
                        _layer_ptr(st_ref, st_addrs, st_strides, i_l, bnd) + head_hi,
                        _layer_ptr(lo_ref, lo_addrs, lo_strides, i_l, bnd) + head_lo,
                        _layer_ptr(sc_ref, sc_addrs, sc_strides, i_l, bnd) + head_sc,
                        b_hi,
                        b_lo,
                        o_v,
                        mask_v,
                        hi_lane,
                        lo_lane,
                        N_HI,
                        N_LO,
                    )
                else:
                    _store_plain(
                        _layer_ptr(st_ref, st_addrs, st_strides, i_l, bnd) + head,
                        b_h,
                        o_v,
                        o_k,
                        mask_h,
                        K,
                    )

    if DAMP:
        _store_damp(
            _layer_ptr(st_ref, st_addrs, st_strides, i_l, dst) + head_hi,
            _layer_ptr(lo_ref, lo_addrs, lo_strides, i_l, dst) + head_lo,
            _layer_ptr(sc_ref, sc_addrs, sc_strides, i_l, dst) + head_sc,
            b_hi,
            b_lo,
            o_v,
            mask_v,
            hi_lane,
            lo_lane,
            N_HI,
            N_LO,
        )
    else:
        _store_plain(
            _layer_ptr(st_ref, st_addrs, st_strides, i_l, dst) + head,
            b_h,
            o_v,
            o_k,
            mask_h,
            K,
        )


def _addrs(ts: Sequence[torch.Tensor], device) -> torch.Tensor:
    return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=device)


def _strides(ts: Sequence[torch.Tensor], dim: int, device) -> torch.Tensor:
    return torch.tensor([t.stride(dim) for t in ts], dtype=torch.int64, device=device)


@dataclass
class GDNRecoverSSMCommitContext:
    conv_states: tuple[torch.Tensor, ...]
    conv_addrs: torch.Tensor
    conv_block_strides: torch.Tensor
    conv_dim_strides: torch.Tensor
    conv_token_strides: torch.Tensor
    conv_history_len: int
    planes: tuple[tuple[torch.Tensor, ...], ...]  # per plane: per-layer tensors
    plane_addrs: tuple[torch.Tensor, ...]
    plane_strides: tuple[torch.Tensor, ...]
    damp: bool
    n_hi: int
    n_lo: int
    num_heads: int
    key_dim: int
    value_dim: int
    spec_query_len: int
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor

    @classmethod
    def create(
        cls, layers: Sequence[Any], *, spec_query_len: int, max_num_reqs: int
    ) -> "GDNRecoverSSMCommitContext":
        if not layers or any(len(layer.kv_cache) != 5 for layer in layers):
            raise ValueError(
                "GDN RecoverSSM pages must hold conv, checkpoint and 3 record tensors"
            )
        conv = [layer.kv_cache[0] for layer in layers]
        if not is_conv_state_dim_first():
            conv = [c.transpose(-1, -2) for c in conv]
        ckpt = [layer.kv_cache[1] for layer in layers]
        rc = [layer.kv_cache[2] for layer in layers]
        rk = [layer.kv_cache[3] for layer in layers]
        rd = [layer.kv_cache[4] for layer in layers]
        num_heads, rec_s, value_dim = rc[0].shape[1:]
        key_dim = rk[0].shape[-1]
        # ponytail: tl.arange(0, K) needs a power of two; GDN K is 128 on
        # every Qwen3.5 checkpoint.
        if key_dim != 128:
            raise ValueError(f"GDN RecoverSSM commit needs K == 128, got {key_dim}")
        # The kernel indexes records with contiguous (HV, S, ...) inner dims.
        for c, k, d in zip(rc, rk, rd):
            if (
                c.shape[1:] != (num_heads, spec_query_len, value_dim)
                or k.shape[1:] != (num_heads, spec_query_len, key_dim)
                or d.shape[1:] != (num_heads, spec_query_len)
                or c.stride()[1:] != (spec_query_len * value_dim, value_dim, 1)
                or k.stride()[1:] != (spec_query_len * key_dim, key_dim, 1)
                or d.stride()[1:] != (spec_query_len, 1)
                or not c.dtype == k.dtype == d.dtype == torch.float32
            ):
                raise ValueError(
                    f"GDN RecoverSSM records need fp32 (HV={num_heads}, "
                    f"S={spec_query_len}, V={value_dim}/K={key_dim}), contiguous"
                )
        device = ckpt[0].device
        damp = ckpt[0].dtype == torch.uint8
        if damp:
            from vllm.model_executor.layers.mamba import damp_runtime

            if value_dim % 32:
                raise ValueError("GDN RecoverSSM DAMP commit needs V % 32 == 0")
            damp_runtime._ensure(device, num_heads)
            views = [damp_runtime._views(p) for p in ckpt]
            hi = [v[0] for v in views]
            lo = [v[1] for v in views]
            sc = [v[2] for v in views]
            n_hi, n_lo = damp_runtime._n_hi, damp_runtime._n_lo
        else:
            hi, lo, sc = ckpt, ckpt, ckpt  # lo/sc unused for plain
            n_hi, n_lo = 0, 0
            for p in ckpt:
                if p.shape[1:] != (num_heads, value_dim, key_dim) or p.stride()[1:] != (
                    value_dim * key_dim,
                    key_dim,
                    1,
                ):
                    raise ValueError("GDN RecoverSSM plain checkpoint layout mismatch")
        planes = (tuple(hi), tuple(lo), tuple(sc), tuple(rc), tuple(rk), tuple(rd))
        conv_len = conv[0].shape[2]
        conv_history_len = conv_len - spec_query_len + 1
        if conv_history_len <= 0:
            raise ValueError("GDN RecoverSSM conv state is shorter than its window")

        def ints():
            return torch.empty(max_num_reqs, dtype=torch.int32, device=device)

        return cls(
            conv_states=tuple(conv),
            conv_addrs=_addrs(conv, device),
            conv_block_strides=_strides(conv, 0, device),
            conv_dim_strides=_strides(conv, 1, device),
            conv_token_strides=_strides(conv, 2, device),
            conv_history_len=conv_history_len,
            planes=planes,
            plane_addrs=tuple(_addrs(p, device) for p in planes),
            plane_strides=tuple(_strides(p, 0, device) for p in planes),
            damp=damp,
            n_hi=n_hi,
            n_lo=n_lo,
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            spec_query_len=spec_query_len,
            commit_lens=ints(),
            final_state_indices=ints(),
            boundary_state_indices=ints(),
            boundary_recovery_lens=ints(),
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
    ) -> None:
        """Replay each request's accepted tokens into its final (and boundary)
        slot, and compact the conv window the same way."""
        batch = state_indices.shape[0]
        if batch == 0:
            return
        if batch > self.commit_lens.shape[0]:
            raise ValueError("GDN RecoverSSM commit batch exceeds its plan capacity")
        if query_start_loc.shape[0] != batch + 1:
            raise ValueError("GDN RecoverSSM commit metadata is incompatible")
        align = block_table is not None
        if block_table is not None:
            if num_computed_tokens is None or mamba_block_size is None:
                raise ValueError("GDN RecoverSSM align metadata is incomplete")
            if mamba_block_size < self.spec_query_len:
                raise ValueError("GDN RecoverSSM align block must cover one window")
        bt_stride = (0, 0) if block_table is None else block_table.stride()
        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            mamba_block_size or 1,
            block_table.shape[1] if block_table is not None else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            bt_stride[0],
            bt_stride[1],
            num_computed_tokens.stride(0) if num_computed_tokens is not None else 0,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )
        num_layers = len(self.conv_states)
        conv_dim = self.conv_states[0].shape[1]
        _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, num_layers)](
            self.conv_states[0],
            self.conv_addrs,
            self.conv_block_strides,
            self.conv_dim_strides,
            self.conv_token_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            conv_dim,
            self.conv_history_len,
            state_indices.stride(0),
            BLOCK_D=256,
            BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len),
            ALIGN_MODE=align,
            num_warps=4,
        )
        bv = min(triton.next_power_of_2(self.value_dim), 32)
        plane_args = []
        for plane, addrs, strides in zip(
            self.planes, self.plane_addrs, self.plane_strides
        ):
            plane_args += [plane[0], addrs, strides]
        grid = (triton.cdiv(self.value_dim, bv), batch, num_layers * self.num_heads)
        _gdn_commit_kernel[grid](
            *plane_args,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            state_indices.stride(0),
            HV=self.num_heads,
            K=self.key_dim,
            V=self.value_dim,
            S=self.spec_query_len,
            BV=bv,
            N_HI=self.n_hi,
            N_LO=self.n_lo,
            N_TILES=self.value_dim // 32,
            DAMP=self.damp,
            ALIGN_MODE=align,
            num_warps=4,
        )


__all__ = ["GDNRecoverSSMCommitContext"]
