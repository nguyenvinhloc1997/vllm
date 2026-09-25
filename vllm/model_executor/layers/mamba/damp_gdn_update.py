# SPDX-License-Identifier: Apache-2.0
"""GDN decode update that reads and writes the packed DAMP page in place.

State columns stay in packed order: the fp16 channels, then the int8
channels. Query and key are packed into that order once per launch, so each
token loads two contiguous spans. The int8 span is quantized in this kernel,
one scale per 32 value rows.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.mamba.damp_runtime import _ensure, _views
from vllm.triton_utils import tl, triton


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_hi"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def _damp_update_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    q,
    k,
    v,
    o,
    h0_hi,
    h0_lo,
    h0_scale,
    stride_hi,
    stride_lo,
    stride_scale,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    N_HI: tl.constexpr,
    N_LO: tl.constexpr,
    N_TILES: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        return

    pos = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = pos < K
    mask_v = o_v < V
    p_q = q + (bos * HV + i_hv) * K
    p_k = k + (bos * HV + i_hv) * K
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    hi_lane = tl.arange(0, 32)
    lo_lane = tl.arange(0, 128)
    b_hi = tl.zeros([BV, 32], dtype=tl.float32)
    b_lo = tl.zeros([BV, 128], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_SPEC_DECODING:
            i_t0 = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
        else:
            i_t0 = 0
        idx_in_row = (i_t0 >= 0) & (i_t0 < stride_indices_seq)
        state_idx = tl.load(
            ssm_state_indices + i_n * stride_indices_seq + i_t0,
            mask=idx_in_row,
            other=0,
        ).to(tl.int64)
        if state_idx <= 0:
            zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
            for _ in range(0, T):
                tl.store(p_o, zero, mask=mask_v)
                p_o += HV * V
            return
        b_scale = tl.load(
            h0_scale + state_idx * stride_scale + i_hv * N_TILES + i_v
        ).to(tl.float32)
        if N_HI > 0:
            b_hi = tl.load(
                h0_hi
                + state_idx * stride_hi
                + (i_hv * V + o_v[:, None]) * N_HI
                + hi_lane[None, :],
                mask=mask_v[:, None] & (hi_lane[None, :] < N_HI),
                other=0,
            ).to(tl.float32)
        if N_LO > 0:
            b_lo = tl.load(
                h0_lo
                + state_idx * stride_lo
                + (i_hv * V + o_v[:, None]) * N_LO
                + lo_lane[None, :],
                mask=mask_v[:, None] & (lo_lane[None, :] < N_LO),
                other=0,
            ).to(tl.float32) * b_scale

    for i_t in range(0, T):
        b_q_hi = tl.load(p_q + hi_lane, mask=hi_lane < N_HI, other=0).to(tl.float32)
        b_k_hi = tl.load(p_k + hi_lane, mask=hi_lane < N_HI, other=0).to(tl.float32)
        b_q_lo = tl.load(p_q + N_HI + lo_lane, mask=lo_lane < N_LO, other=0).to(tl.float32)
        b_k_lo = tl.load(p_k + N_HI + lo_lane, mask=lo_lane < N_LO, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
        softplus_x = tl.where(
            beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
        )
        b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x
        b_beta = tl.sigmoid(b_b.to(tl.float32))
        if USE_QK_L2NORM_IN_KERNEL:
            qn = tl.rsqrt(tl.sum(b_q_hi * b_q_hi) + tl.sum(b_q_lo * b_q_lo) + 1e-6)
            kn = tl.rsqrt(tl.sum(b_k_hi * b_k_hi) + tl.sum(b_k_lo * b_k_lo) + 1e-6)
            b_q_hi = b_q_hi * qn
            b_q_lo = b_q_lo * qn
            b_k_hi = b_k_hi * kn
            b_k_lo = b_k_lo * kn
        b_q_hi = b_q_hi * scale
        b_q_lo = b_q_lo * scale
        decay = tl.exp(b_g)
        b_hi *= decay
        b_lo *= decay
        b_v -= tl.sum(b_hi * b_k_hi[None, :], 1)
        b_v -= tl.sum(b_lo * b_k_lo[None, :], 1)
        b_v *= b_beta
        b_hi += b_v[:, None] * b_k_hi[None, :]
        b_lo += b_v[:, None] * b_k_lo[None, :]
        b_o = tl.sum(b_hi * b_q_hi[None, :], 1) + tl.sum(b_lo * b_q_lo[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        final_state_idx = tl.load(
            ssm_state_indices + i_n * stride_indices_seq + i_t
        ).to(tl.int64)
        if final_state_idx > 0:
            if N_HI > 0:
                tl.store(
                    h0_hi
                    + final_state_idx * stride_hi
                    + (i_hv * V + o_v[:, None]) * N_HI
                    + hi_lane[None, :],
                    b_hi.to(tl.float16),
                    mask=mask_v[:, None] & (hi_lane[None, :] < N_HI),
                )
            if N_LO > 0:
                amax = tl.max(tl.abs(b_lo))
                sc = tl.maximum(amax, 1e-8) / 127.0
                qv = b_lo / sc
                codes = tl.where(qv >= 0, tl.floor(qv + 0.5), tl.ceil(qv - 0.5))
                codes = tl.minimum(tl.maximum(codes, -127.0), 127.0)
                tl.store(
                    h0_lo
                    + final_state_idx * stride_lo
                    + (i_hv * V + o_v[:, None]) * N_LO
                    + lo_lane[None, :],
                    codes.to(tl.int8),
                    mask=mask_v[:, None] & (lo_lane[None, :] < N_LO),
                )
                tl.store(
                    h0_scale + final_state_idx * stride_scale + i_hv * N_TILES + i_v,
                    sc,
                )

        p_q += HV * K
        p_k += HV * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV


def damp_fused_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):
    """Same call as the fp16 update. ``initial_state`` is the uint8 page."""
    if is_kda:
        raise RuntimeError("DAMP packed order is for one decay scalar per head")
    if not inplace_final_state:
        raise RuntimeError("DAMP update stores through the packed page")
    if initial_state is None or ssm_state_indices is None:
        raise RuntimeError("DAMP update requires the packed page and state indices")
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    if NK != 1:
        raise RuntimeError("NK > 1 is not supported")
    if scale is None:
        scale = K**-0.5
    _ensure(initial_state.device, HV)
    from vllm.model_executor.layers.mamba import damp_runtime as rt
    from vllm.model_executor.layers.mamba.damp_pack import pack_qk

    hi, lo, sc = _views(initial_state)
    n_hi = rt._n_hi
    n_lo = rt._n_lo
    q = pack_qk(q, rt._hi_idx, rt._lo_idx, HV)
    k = pack_qk(k, rt._hi_idx, rt._lo_idx, HV)
    o = q.new_empty(NK, *v.shape)
    if ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()
    _damp_update_kernel[(NK, NV, N * HV)](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        o=o,
        h0_hi=hi,
        h0_lo=lo,
        h0_scale=sc,
        stride_hi=hi.stride(0),
        stride_lo=lo.stride(0),
        stride_scale=sc.stride(0),
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        N_HI=n_hi,
        N_LO=n_lo,
        N_TILES=4,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=4,
        num_stages=3,
    )
    return o.squeeze(0), initial_state
