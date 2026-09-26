# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/compress.py
"""Prefill KV compress orchestration for H2O (Algorithm 1).

Ownership-A block-table resize / free-unused is deferred; this module selects
and gathers K/V (+ absolute RoPE positions) only.
"""

from __future__ import annotations

import torch

from vllm import envs
from vllm.v1.h2o.policy import H2OState, compute_k, select_prefill
from vllm.v1.h2o.scores import accumulate_attention_mass


def should_compress_prefill(
    *, is_full_attention: bool, is_last_prefill_chunk: bool
) -> bool:
    """True when end-of-prefill H2O compress should run for this layer/step."""
    return bool(envs.VLLM_H2O) and is_full_attention and is_last_prefill_chunk


def compress_prefill_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: list[int],
    ratio: float = 0.2,
) -> tuple[H2OState, torch.Tensor, torch.Tensor, list[int]]:
    """Score, select, and gather prefill K/V keeping absolute positions.

    Args:
        q: Query tensor ``[T, H_q, D]``.
        k: Key tensor ``[T, H_kv, D]``.
        v: Value tensor ``[T, H_kv, D]``.
        positions: Absolute RoPE positions length ``T`` (not necessarily 0..T-1).
        ratio: Total keep fraction (even split into heavy + recent).

    Returns:
        ``(state, k_out, v_out, positions_out)`` where ``positions_out`` is the
        kept absolute positions in causal order, and ``state.heavy`` /
        ``state.recent`` use those same absolute positions.
    """
    t = k.shape[0]
    if len(positions) != t:
        raise ValueError(f"positions length {len(positions)} != key length {t}")
    if q.shape[0] != t or v.shape[0] != t:
        raise ValueError("q, k, v must share the same sequence length")

    scale = q.shape[-1] ** -0.5
    mass = accumulate_attention_mass(q, k, scale=scale)
    local_scores = {i: float(mass[i].item()) for i in range(t)}
    k_budget = compute_k(t, ratio)
    local_state = select_prefill(local_scores, t, k_budget)

    heavy = [positions[i] for i in local_state.heavy]
    recent = [positions[i] for i in local_state.recent]
    scores = {positions[i]: s for i, s in local_state.scores.items()}
    state = H2OState(k=local_state.k, heavy=heavy, recent=recent, scores=scores)

    keep_local = sorted(set(local_state.heavy) | set(local_state.recent))
    positions_out = [positions[i] for i in keep_local]
    return state, k[keep_local], v[keep_local], positions_out


def try_compress_prefill_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: list[int],
    *,
    is_full_attention: bool,
    is_last_prefill_chunk: bool,
    ratio: float | None = None,
) -> tuple[H2OState, torch.Tensor, torch.Tensor, list[int]] | None:
    """Flag-gated wrapper around :func:`compress_prefill_kv`.

    Returns ``None`` when H2O is off, the layer is not full-attention, or this
    is not the last prefill chunk. Does not resize block tables (Ownership-A
    deferred).
    """
    if not should_compress_prefill(
        is_full_attention=is_full_attention,
        is_last_prefill_chunk=is_last_prefill_chunk,
    ):
        return None
    if ratio is None:
        ratio = float(envs.VLLM_H2O_RATIO)
    return compress_prefill_kv(q, k, v, positions, ratio=ratio)


def is_full_attention_sliding_window(
    sliding_window: tuple[int, int] | None,
) -> bool:
    """True for global / full-attention layers (no finite sliding window)."""
    if sliding_window is None:
        return True
    return sliding_window[0] < 0 and sliding_window[1] < 0


def after_full_attention_kv_update(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor | None = None,
    positions: list[int] | None = None,
    is_last_prefill_chunk: bool = False,
    sliding_window: tuple[int, int] | None = None,
    ratio: float | None = None,
) -> tuple[H2OState, torch.Tensor, torch.Tensor, list[int]] | None:
    """Hook (b) entry: after full-attention KV pages are written.

    When the runner supplies end-of-prefill ``query`` + absolute ``positions``
    and ``is_last_prefill_chunk``, runs :func:`try_compress_prefill_kv`.
    Ownership-A write-back / block resize is not applied here.
    """
    if query is None or positions is None:
        return None
    return try_compress_prefill_kv(
        query,
        key,
        value,
        positions,
        is_full_attention=is_full_attention_sliding_window(sliding_window),
        is_last_prefill_chunk=is_last_prefill_chunk,
        ratio=ratio,
    )
