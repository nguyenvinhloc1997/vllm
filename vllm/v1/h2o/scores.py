# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/scores.py
from __future__ import annotations

import torch


def accumulate_attention_mass(
    q: torch.Tensor, k: torch.Tensor, scale: float
) -> torch.Tensor:
    """Dense tiny-T helper for unit tests. Do not use on the production path."""
    # q: [T_q, H_q, D], k: [T_k, H_kv, D]
    t_q, h_q, d = q.shape
    t_k, h_kv, d_k = k.shape
    assert d == d_k and h_q % h_kv == 0
    group = h_q // h_kv
    # expand k to query heads
    k_exp = k.repeat_interleave(group, dim=1)
    # scores [T_q, H_q, T_k]
    logits = torch.einsum("qhd,khd->qhk", q * scale, k_exp)
    if t_q == t_k:
        mask = torch.triu(torch.ones(t_q, t_k, dtype=torch.bool, device=q.device), 1)
        logits = logits.masked_fill(mask.unsqueeze(1), float("-inf"))
    weights = torch.softmax(logits.float(), dim=-1).to(q.dtype)
    # sum queries → [H_q, T_k]; mean GQA group → [H_kv, T_k]; sum heads → [T_k]
    per_q_head = weights.sum(dim=0)
    per_kv = per_q_head.view(h_kv, group, t_k).mean(dim=1)
    return per_kv.sum(dim=0)


def accumulate_attention_mass_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    *,
    tile_q: int = 64,
    tile_k: int = 64,
    q_positions: list[int] | None = None,
    k_positions: list[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """S2 production mass: tile Q vs K; never allocate a ``T_q × T_k`` matrix.

    Two passes per query tile (max/denom, then weighted scatter onto keys).
    When ``t_q == t_k`` and positions are omitted, applies a causal upper-tri
    mask on local indices. When absolute ``q_positions`` / ``k_positions`` are
    given, a query attends only to keys with ``k_pos <= q_pos``.

    Args:
        q: ``[T_q, H_q, D]``.
        k: ``[T_k, H_kv, D]``.
        scale: Softmax scale (typically ``D ** -0.5``).
        tile_q: Query tile length.
        tile_k: Key tile length.
        q_positions: Optional absolute positions length ``T_q``.
        k_positions: Optional absolute positions length ``T_k``.
        out: Optional pre-allocated ``[T_k]`` float32 buffer to add into.

    Returns:
        Per-key attention mass ``[T_k]`` (float32), summed across heads.
    """
    t_q, h_q, d = q.shape
    t_k, h_kv, d_k = k.shape
    if d != d_k or h_q % h_kv != 0:
        raise ValueError(f"bad shapes q={tuple(q.shape)} k={tuple(k.shape)}")
    group = h_q // h_kv
    device = q.device
    mass = (
        out if out is not None else torch.zeros(t_k, device=device, dtype=torch.float32)
    )
    if mass.numel() != t_k:
        raise ValueError(f"out length {mass.numel()} != T_k {t_k}")

    if q_positions is not None and len(q_positions) != t_q:
        raise ValueError("q_positions length mismatch")
    if k_positions is not None and len(k_positions) != t_k:
        raise ValueError("k_positions length mismatch")
    use_abs = q_positions is not None and k_positions is not None
    causal_local = (not use_abs) and t_q == t_k

    q_pos_t = (
        torch.tensor(q_positions, device=device, dtype=torch.int64) if use_abs else None
    )
    k_pos_t = (
        torch.tensor(k_positions, device=device, dtype=torch.int64) if use_abs else None
    )

    for q0 in range(0, t_q, tile_q):
        q1 = min(q0 + tile_q, t_q)
        q_tile = q[q0:q1].float() * scale  # [tq, H_q, D]
        tq = q1 - q0
        m_i = torch.full((tq, h_q), float("-inf"), device=device, dtype=torch.float32)
        l_i = torch.zeros(tq, h_q, device=device, dtype=torch.float32)

        def _logits(
            k0: int,
            k1: int,
            *,
            q_tile: torch.Tensor = q_tile,
            q0: int = q0,
            q1: int = q1,
            tq: int = tq,
        ) -> torch.Tensor:
            k_tile = k[k0:k1].float()
            k_exp = k_tile.repeat_interleave(group, dim=1)
            logits = torch.einsum("qhd,khd->qhk", q_tile, k_exp)
            if use_abs:
                assert q_pos_t is not None and k_pos_t is not None
                q_idx = q_pos_t[q0:q1].view(tq, 1, 1)
                k_idx = k_pos_t[k0:k1].view(1, 1, -1)
                logits = logits.masked_fill(k_idx > q_idx, float("-inf"))
            elif causal_local:
                q_idx = torch.arange(q0, q1, device=device).view(tq, 1, 1)
                k_idx = torch.arange(k0, k1, device=device).view(1, 1, -1)
                logits = logits.masked_fill(k_idx > q_idx, float("-inf"))
            return logits

        # Pass 1: online max + denom.
        for k0 in range(0, t_k, tile_k):
            k1 = min(k0 + tile_k, t_k)
            logits = _logits(k0, k1)
            block_max = logits.amax(dim=-1)
            new_m = torch.maximum(m_i, block_max)
            l_i = l_i * torch.exp(m_i - new_m) + torch.exp(
                logits - new_m.unsqueeze(-1)
            ).sum(dim=-1)
            m_i = new_m

        # Pass 2: scatter normalized weights onto keys.
        for k0 in range(0, t_k, tile_k):
            k1 = min(k0 + tile_k, t_k)
            logits = _logits(k0, k1)
            weights = torch.exp(logits - m_i.unsqueeze(-1)) / l_i.clamp_min(
                1e-12
            ).unsqueeze(-1)
            # weights: [tq, H_q, tk] → sum queries → [H_q, tk]
            per_q_head = weights.sum(dim=0)
            per_kv = per_q_head.view(h_kv, group, k1 - k0).mean(dim=1)
            mass[k0:k1] += per_kv.sum(dim=0)

    return mass


def mass_dict(mass: torch.Tensor, positions: list[int]) -> dict[int, float]:
    assert mass.numel() == len(positions)
    return {int(p): float(m) for p, m in zip(positions, mass.tolist())}
