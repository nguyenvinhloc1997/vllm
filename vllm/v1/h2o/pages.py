# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/pages.py
"""Gather / write helpers for FlashAttention paged KV (bf16 layout)."""

from __future__ import annotations

import torch

from vllm.v1.h2o.rewrite import slot_to_block_offset
from vllm.v1.h2o.slots import SlotLayout


def is_fa_paged_kv_cache(kv_cache: torch.Tensor) -> bool:
    """True for FlashAttention paged layout ``[num_blocks, H_kv, block_size, 2*D]``.

    KVarN packed tiles are typically ``[num_blocks, H_kv, tile_bytes]`` (3-D)
    or a vLLM reinterpretation ``[num_blocks, group, H_kv, tile_bytes]`` (4-D).
    The 4-D form has an even last dim (e.g. 140) and must not be treated as FA
    — FA puts ``H_kv`` before ``block_size``; KVarN puts ``group`` first.

    Treating KVarN as FA sends post-commit scoring into
    ``accumulate_attention_mass`` with mismatched ``D`` and kills the engine
    (Task 15 compress-eligible ``AssertionError`` / HTTP 500).
    """
    if not isinstance(kv_cache, torch.Tensor) or kv_cache.ndim != 4:
        return False
    last = int(kv_cache.shape[-1])
    if last < 2 or last % 2 != 0:
        return False
    # Non-floating caches are packed tiles (KVarN), never FA bf16/fp16 pages.
    if not kv_cache.is_floating_point():
        return False
    head_size = last // 2
    # FA ``2*D`` uses a modest power-of-two head size; KVarN ``tile_bytes/2``
    # is not (e.g. 70 for tile=140, or thousands for full tiles).
    if head_size < 8 or head_size > 512 or (head_size & (head_size - 1)) != 0:
        return False
    # FA: [B, H_kv, block_size, 2D] — heads (small) before block_size.
    # KVarN reinterpret: [B, group, H_kv, tile] — group (>=64) before heads.
    dim1, dim2 = int(kv_cache.shape[1]), int(kv_cache.shape[2])
    return not (dim1 >= 64 and dim1 > dim2)


def split_fa_kv_cache(
    kv_cache: torch.Tensor, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split packed FA cache into key/value views.

    Args:
        kv_cache: ``[num_blocks, H_kv, block_size, 2*D]``.
        head_size: ``D``.
    """
    return kv_cache.transpose(1, 2).split(head_size, dim=-1)


def head_size_from_fa_kv_cache(kv_cache: torch.Tensor) -> int:
    return int(kv_cache.shape[-1] // 2)


def gather_kv_from_slots(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: list[int],
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read K/V for physical slots → ``[T, H_kv, D]`` each."""
    if not slots:
        h = key_cache.shape[2]
        d = key_cache.shape[3]
        empty_k = key_cache.new_empty(0, h, d)
        empty_v = value_cache.new_empty(0, h, d)
        return empty_k, empty_v
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for slot in slots:
        bid, off = slot_to_block_offset(int(slot), block_size)
        keys.append(key_cache[bid, off])
        values.append(value_cache[bid, off])
    return torch.stack(keys, dim=0), torch.stack(values, dim=0)


class _FaKvGather:
    def __init__(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> None:
        self._key_cache = key_cache
        self._value_cache = value_cache

    def gather_kv(
        self,
        *,
        request_index: int,
        slots: list[int],
        seq_len: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del request_index
        return gather_kv_from_slots(
            self._key_cache,
            self._value_cache,
            slots[:seq_len],
            block_size=block_size,
        )


def fa_kv_gather(key_cache: torch.Tensor, value_cache: torch.Tensor) -> _FaKvGather:
    """Return a FlashAttention page gather closed over one layer's cache."""
    return _FaKvGather(key_cache, value_cache)


def gather_kv_for_positions(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    layout: SlotLayout,
    positions: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather retained K/V for absolute positions via :class:`SlotLayout`."""
    slots = [layout.slot_for_pos(int(p)) for p in positions]
    return gather_kv_from_slots(
        key_cache, value_cache, slots, block_size=layout.block_size
    )


def read_slot_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot: int,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one slot → ``[H_kv, D]`` key/value."""
    bid, off = slot_to_block_offset(int(slot), block_size)
    return key_cache[bid, off].clone(), value_cache[bid, off].clone()


def write_slot_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot: int,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_size: int,
) -> None:
    bid, off = slot_to_block_offset(int(slot), block_size)
    key_cache[bid, off].copy_(key.to(dtype=key_cache.dtype))
    value_cache[bid, off].copy_(value.to(dtype=value_cache.dtype))
