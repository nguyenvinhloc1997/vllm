# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/slots.py
"""Ownership-A dense slot layout and decode write remapping.

After prefill compress, kept tokens are packed into the leading
``num_keep`` slots of the truncated FA block table. Stock
``TOKEN_TO_KV_SLOT`` still maps absolute RoPE positions, which walk past
the compressed table on decode — this module remaps decode writes onto
the circular-recent / victim slots inside the 2K layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.v1.h2o.policy import H2OState, decode_step


@dataclass
class SlotLayout:
    """Dense map from absolute positions to physical FA slots."""

    block_ids: list[int]
    block_size: int
    num_keep: int
    # Absolute RoPE position → local index in ``[0, num_keep)``.
    pos_to_local: dict[int, int] = field(default_factory=dict)

    def slot_for_local(self, local: int) -> int:
        if local < 0 or local >= self.num_keep:
            raise IndexError(f"local index {local} outside [0, {self.num_keep})")
        block_idx = local // self.block_size
        if block_idx >= len(self.block_ids):
            raise IndexError(
                f"local {local} needs block {block_idx}, have {len(self.block_ids)}"
            )
        return int(self.block_ids[block_idx]) * self.block_size + (
            local % self.block_size
        )

    def slot_for_pos(self, pos: int) -> int:
        return self.slot_for_local(self.pos_to_local[pos])

    def local_for_slot(self, slot: int) -> int:
        for local in range(self.num_keep):
            if self.slot_for_local(local) == slot:
                return local
        raise KeyError(f"slot {slot} not in layout")


def build_slot_layout(
    positions_out: list[int],
    block_ids: list[int],
    block_size: int,
) -> SlotLayout:
    """Build layout after ``repack_kv_into_pages`` (causal-dense pack)."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    n = len(positions_out)
    need_blocks = (n + block_size - 1) // block_size if n else 0
    if len(block_ids) < need_blocks:
        raise ValueError(
            f"need {need_blocks} blocks for {n} tokens, got {len(block_ids)}"
        )
    pos_to_local = {int(p): i for i, p in enumerate(positions_out)}
    if len(pos_to_local) != n:
        raise ValueError("positions_out must be unique")
    return SlotLayout(
        block_ids=list(block_ids[:need_blocks]),
        block_size=block_size,
        num_keep=n,
        pos_to_local=pos_to_local,
    )


@dataclass
class DecodeSlotPlan:
    """Physical writes implied by one ``decode_step`` at capacity."""

    new_pos: int
    # Physical slot that receives the new token's K/V (circular recent).
    new_token_slot: int
    # Absolute position whose slot is reused for ``new_pos`` (leaves the set).
    dropped_pos: int
    # If the aged-out recent is promoted into heavy, copy its KV from the
    # circular slot into the evicted heavy's slot before overwrite.
    promote_copy: tuple[int, int] | None  # (src_slot, dst_slot)
    state_after: H2OState


def plan_decode_slot_writes(
    state: H2OState,
    layout: SlotLayout,
    new_pos: int,
    new_scores_delta: dict[int, float],
) -> DecodeSlotPlan:
    """Plan dirty-slot writes for one committed decode token.

    At capacity (``len(heavy)==k`` and ``len(recent)==k``), the new token
    always lands in the aged-out recent's physical slot (paper circular
    recent). The unique dropped position frees a slot for ``new_pos``;
    when the aged-out token is promoted into heavy, its KV is copied into
    the evicted heavy's slot first.
    """
    if state.k <= 0:
        raise ValueError("empty H2O budget")
    at_capacity = len(state.recent) == state.k and len(state.heavy) == state.k
    state_after = decode_step(state, new_pos, new_scores_delta)

    if not at_capacity:
        local = layout.num_keep
        block_idx = local // layout.block_size
        if block_idx >= len(layout.block_ids):
            raise ValueError("no free local slot to append decode token")
        slot = int(layout.block_ids[block_idx]) * layout.block_size + (
            local % layout.block_size
        )
        return DecodeSlotPlan(
            new_pos=new_pos,
            new_token_slot=slot,
            dropped_pos=-1,
            promote_copy=None,
            state_after=state_after,
        )

    old_recent = state.recent[0]
    before = set(state.heavy) | set(state.recent)
    after = set(state_after.heavy) | set(state_after.recent)
    dropped = before - after
    if len(dropped) != 1:
        raise RuntimeError(f"expected one drop, got {dropped}")
    dropped_pos = next(iter(dropped))

    if old_recent in after and dropped_pos != old_recent:
        src = layout.slot_for_pos(old_recent)
        dst = layout.slot_for_pos(dropped_pos)
        promote_copy: tuple[int, int] | None = (src, dst)
        new_slot = src
    elif dropped_pos == old_recent:
        promote_copy = None
        new_slot = layout.slot_for_pos(old_recent)
    else:
        raise RuntimeError(
            f"unexpected drop {dropped_pos} (old_recent={old_recent}, after={after})"
        )

    return DecodeSlotPlan(
        new_pos=new_pos,
        new_token_slot=new_slot,
        dropped_pos=dropped_pos,
        promote_copy=promote_copy,
        state_after=state_after,
    )


def apply_slot_layout_after_decode(
    layout: SlotLayout,
    plan: DecodeSlotPlan,
) -> SlotLayout:
    """Return an updated layout after applying ``plan`` (CPU bookkeeping)."""
    pos_to_local = dict(layout.pos_to_local)
    if plan.dropped_pos < 0:
        local = layout.num_keep
        pos_to_local[plan.new_pos] = local
        return SlotLayout(
            block_ids=list(layout.block_ids),
            block_size=layout.block_size,
            num_keep=layout.num_keep + 1,
            pos_to_local=pos_to_local,
        )

    if plan.promote_copy is None:
        local = pos_to_local.pop(plan.dropped_pos)
        pos_to_local[plan.new_pos] = local
        return SlotLayout(
            block_ids=list(layout.block_ids),
            block_size=layout.block_size,
            num_keep=layout.num_keep,
            pos_to_local=pos_to_local,
        )

    src_slot, dst_slot = plan.promote_copy
    src_local = layout.local_for_slot(src_slot)
    dst_local = layout.local_for_slot(dst_slot)
    pos_to_local.pop(plan.dropped_pos, None)
    aged_pos = None
    for p, loc in pos_to_local.items():
        if loc == src_local:
            aged_pos = p
            break
    if aged_pos is None:
        raise RuntimeError("aged recent missing from layout")
    pos_to_local[aged_pos] = dst_local
    pos_to_local[plan.new_pos] = src_local
    return SlotLayout(
        block_ids=list(layout.block_ids),
        block_size=layout.block_size,
        num_keep=layout.num_keep,
        pos_to_local=pos_to_local,
    )


def remap_decode_write_slot(
    state: H2OState,
    layout: SlotLayout,
) -> int:
    """Physical slot for the next decode KV write (circular recent head).

    Used to patch ``slot_mapping`` before the forward so stock FA writes
    into the compressed 2K table instead of an absolute-position OOB slot.
    Does not mutate state; eviction bookkeeping runs after commit.
    """
    if state.k <= 0:
        raise ValueError("empty H2O budget")
    if len(state.recent) < state.k:
        local = len(set(state.heavy) | set(state.recent))
        if local < layout.num_keep:
            return layout.slot_for_local(local)
        return layout.slot_for_local(max(layout.num_keep - 1, 0))
    return layout.slot_for_pos(state.recent[0])
