# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/rewrite.py
"""Dirty-page KV rewrites for H2O decode (bf16 FA; KVarN-pluggable).

Only victim / new-recent slots are touched — never a full 2K repack.
Upstream extract can swap :func:`rewrite_dirty_slots` for packed-tile
writes without changing :mod:`vllm.v1.h2o.policy`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.h2o.slots import DecodeSlotPlan, SlotLayout


@dataclass
class DirtySlotWrite:
    """One physical slot overwrite (K and V, FA layout)."""

    slot: int
    key: torch.Tensor  # [H_kv, D]
    value: torch.Tensor  # [H_kv, D]


def slot_to_block_offset(slot: int, block_size: int) -> tuple[int, int]:
    return int(slot) // block_size, int(slot) % block_size


def rewrite_dirty_slots(
    *,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_size: int,
    writes: list[DirtySlotWrite],
) -> None:
    """Overwrite individual FA slots in-place (bf16-friendly).

    Args:
        key_cache: ``[num_blocks, block_size, H_kv, D]`` (FA split layout).
        value_cache: Same shape as ``key_cache``.
        block_size: Tokens per block (must match cache dim 1).
        writes: Slots to replace; order is applied sequentially so a
            promote-copy can land before the circular recent overwrite.
    """
    if not writes:
        return
    if key_cache.shape[1] != block_size or value_cache.shape[1] != block_size:
        raise ValueError(
            f"block_size {block_size} != cache dim1 "
            f"{key_cache.shape[1]}/{value_cache.shape[1]}"
        )
    for w in writes:
        bid, off = slot_to_block_offset(w.slot, block_size)
        if w.key.shape != w.value.shape:
            raise ValueError("key/value shape mismatch")
        key_cache[bid, off].copy_(w.key.to(dtype=key_cache.dtype))
        value_cache[bid, off].copy_(w.value.to(dtype=value_cache.dtype))


def copy_slot(
    *,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_size: int,
    src_slot: int,
    dst_slot: int,
) -> None:
    """Copy one FA slot to another (promote aged-out recent into heavy)."""
    if src_slot == dst_slot:
        return
    s_bid, s_off = slot_to_block_offset(src_slot, block_size)
    d_bid, d_off = slot_to_block_offset(dst_slot, block_size)
    key_cache[d_bid, d_off].copy_(key_cache[s_bid, s_off])
    value_cache[d_bid, d_off].copy_(value_cache[s_bid, s_off])


def apply_decode_plan_rewrites(
    *,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    layout: SlotLayout,
    plan: DecodeSlotPlan,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
) -> None:
    """Execute promote-copy (if any) then write the new token into its slot.

    Structured so a KVarN backend can replace the bodies of
    :func:`copy_slot` / :func:`rewrite_dirty_slots` without touching policy.
    """
    if plan.promote_copy is not None:
        src, dst = plan.promote_copy
        copy_slot(
            key_cache=key_cache,
            value_cache=value_cache,
            block_size=layout.block_size,
            src_slot=src,
            dst_slot=dst,
        )
    rewrite_dirty_slots(
        key_cache=key_cache,
        value_cache=value_cache,
        block_size=layout.block_size,
        writes=[DirtySlotWrite(slot=plan.new_token_slot, key=new_key, value=new_value)],
    )
