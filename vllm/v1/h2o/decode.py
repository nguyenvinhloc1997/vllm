# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/decode.py
"""Committed-token decode maintenance (paper Algorithm 1).

Scores only committed queries. Does not change DFlash ``num_query_per_req``
or CUDA-graph capture shapes — orchestration runs after
``Scheduler.update_from_output`` / ``EngineCore.post_step``.
"""

from __future__ import annotations

import torch

from vllm.v1.h2o.policy import H2OState
from vllm.v1.h2o.rewrite import apply_decode_plan_rewrites
from vllm.v1.h2o.runtime import H2OLayerRuntime, H2ORequestRuntime
from vllm.v1.h2o.scores import accumulate_attention_mass, mass_dict
from vllm.v1.h2o.slots import (
    DecodeSlotPlan,
    SlotLayout,
    apply_slot_layout_after_decode,
    plan_decode_slot_writes,
    remap_decode_write_slot,
)


def score_committed_decode_delta(
    q_committed: torch.Tensor,
    k_retained: torch.Tensor,
    retained_positions: list[int],
    *,
    scale: float | None = None,
) -> dict[int, float]:
    """Attention mass from committed queries onto the retained key set.

    Args:
        q_committed: ``[T_q, H_q, D]`` — accepted tokens only (never drafts).
        k_retained: ``[T_k, H_kv, D]`` — heavy ∪ recent keys.
        retained_positions: Absolute positions length ``T_k``.
    """
    if q_committed.shape[0] == 0:
        return {}
    if scale is None:
        scale = q_committed.shape[-1] ** -0.5
    mass = accumulate_attention_mass(q_committed, k_retained, scale=scale)
    return mass_dict(mass, retained_positions)


def apply_committed_decode_step(
    layer_rt: H2OLayerRuntime,
    *,
    new_pos: int,
    new_scores_delta: dict[int, float],
    key_cache: torch.Tensor | None = None,
    value_cache: torch.Tensor | None = None,
    new_key: torch.Tensor | None = None,
    new_value: torch.Tensor | None = None,
) -> DecodeSlotPlan:
    """Run policy ``decode_step``, update layout, optionally rewrite dirty slots."""
    plan = plan_decode_slot_writes(
        layer_rt.state, layer_rt.layout, new_pos, new_scores_delta
    )
    if (
        key_cache is not None
        and value_cache is not None
        and new_key is not None
        and new_value is not None
    ):
        apply_decode_plan_rewrites(
            key_cache=key_cache,
            value_cache=value_cache,
            layout=layer_rt.layout,
            plan=plan,
            new_key=new_key,
            new_value=new_value,
        )
    layer_rt.layout = apply_slot_layout_after_decode(layer_rt.layout, plan)
    layer_rt.state = plan.state_after
    return plan


def next_decode_write_slot(layer_rt: H2OLayerRuntime) -> int:
    """Slot to patch into ``slot_mapping`` for the upcoming decode write."""
    return remap_decode_write_slot(layer_rt.state, layer_rt.layout)


def retained_positions(state: H2OState) -> list[int]:
    return sorted(set(state.heavy) | set(state.recent))


def apply_pending_committed_steps_cpu(
    req_rt: H2ORequestRuntime,
    *,
    layer_name: str,
    deltas_per_pos: list[dict[int, float]],
) -> list[H2OState]:
    """Apply pending committed positions with precomputed score deltas (CPU).

    Used by unit tests and by the gated post-step hook when the worker
    supplies per-token mass dicts. Clears ``pending_committed_positions``.
    """
    layer_rt = req_rt.layers[layer_name]
    positions = list(req_rt.pending_committed_positions)
    if len(deltas_per_pos) != len(positions):
        raise ValueError(
            f"deltas ({len(deltas_per_pos)}) != positions ({len(positions)})"
        )
    states: list[H2OState] = []
    for pos, delta in zip(positions, deltas_per_pos):
        apply_committed_decode_step(layer_rt, new_pos=pos, new_scores_delta=delta)
        states.append(layer_rt.state)
    req_rt.pending_committed_positions.clear()
    return states


# Re-export layout type for callers.
__all__ = [
    "SlotLayout",
    "score_committed_decode_delta",
    "apply_committed_decode_step",
    "next_decode_write_slot",
    "retained_positions",
    "apply_pending_committed_steps_cpu",
]
