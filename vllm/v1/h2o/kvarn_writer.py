# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# KVarN KeptKvWriter: chunked tile store with per-tile pool reclaim.
# Live image may overwrite this via kvarn/install.sh from hyperqwen-ngm.

"""Pack selected H2O K/V into KVarN pages (fresh retained block ids)."""

from __future__ import annotations

import torch


def _ensure_one_pool_slot(impl, block_id: int, device: torch.device) -> None:
    """Allocate one fp16 pool slot for ``block_id`` if missing."""
    cls = type(impl)
    gk = impl._group_key
    bid = int(block_id)
    impl._ensure_pool(device, num_blocks_hint=bid + 1)
    mkey = (device, gk)
    dict_map = cls._block_to_slot_dict[gk]
    free_slots = cls._free_slots[gk]
    b2s_t = cls._block_to_slot_t_per_device[mkey]
    if bid in dict_map:
        return
    if not free_slots:
        raise RuntimeError(
            f"KVarN pool exhausted during H2O pack (group={gk}, need block {bid})"
        )
    slot = free_slots.pop()
    dict_map[bid] = slot
    if bid < b2s_t.shape[0]:
        b2s_t[bid] = slot
    cls._max_known_block_id[gk] = max(cls._max_known_block_id.get(gk, 0), bid)


def _release_pool_slot(impl, block_id: int, device: torch.device) -> None:
    """Return one block's pool slot after a successful flush (builder-style)."""
    cls = type(impl)
    gk = impl._group_key
    bid = int(block_id)
    dict_map = cls._block_to_slot_dict[gk]
    free_slots = cls._free_slots[gk]
    mkey = (device, gk)
    b2s_t = cls._block_to_slot_t_per_device[mkey]
    slot = dict_map.pop(bid, None)
    if slot is None:
        return
    free_slots.append(slot)
    if bid < b2s_t.shape[0]:
        b2s_t[bid] = -1


def write_kept_kv_chunked(
    impl,
    kv_cache: torch.Tensor,
    layer,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    *,
    block_ids: list[int],
    block_size: int,
) -> int:
    """Write kept K/V into ``block_ids`` one tile at a time.

    Full tiles: ensure slot → ``do_kv_cache_update`` → ``_batched_flush`` →
    release slot. Partial last tile: ensure + update only (stays in fp16 pool).

    Returns:
        Number of block ids touched (including a partial tail block).
    """
    t_keep = int(k_out.shape[0])
    if t_keep == 0:
        return 0
    n_blocks = (t_keep + block_size - 1) // block_size
    if len(block_ids) < n_blocks:
        raise ValueError(
            f"need {n_blocks} blocks to pack {t_keep} tokens, got {len(block_ids)}"
        )
    bids = [int(b) for b in block_ids[:n_blocks]]
    device = k_out.device

    key = k_out.reshape(t_keep, -1).contiguous()
    value = v_out.reshape(t_keep, -1).contiguous()
    if key.dtype != torch.float16:
        key = key.to(torch.float16)
        value = value.to(torch.float16)

    n_full = t_keep // block_size
    rem = t_keep % block_size

    for bi in range(n_full):
        bid = bids[bi]
        _ensure_one_pool_slot(impl, bid, device)
        s0 = bi * block_size
        s1 = s0 + block_size
        slots = torch.arange(
            bid * block_size,
            bid * block_size + block_size,
            dtype=torch.long,
            device=device,
        )
        impl.do_kv_cache_update(
            layer=layer,
            key=key[s0:s1],
            value=value[s0:s1],
            kv_cache=kv_cache,
            slot_mapping=slots,
        )
        type(impl)._batched_flush([(impl, bid, kv_cache)])
        _release_pool_slot(impl, bid, device)

    if rem:
        bid = bids[n_full]
        _ensure_one_pool_slot(impl, bid, device)
        s0 = n_full * block_size
        slots = torch.arange(
            bid * block_size,
            bid * block_size + rem,
            dtype=torch.long,
            device=device,
        )
        impl.do_kv_cache_update(
            layer=layer,
            key=key[s0:],
            value=value[s0:],
            kv_cache=kv_cache,
            slot_mapping=slots,
        )
        # Partial tile stays resident in the fp16 pool (no flush).

    return n_blocks


class _KvarnKeptKvWriter:
    def __init__(self, impl, kv_cache: torch.Tensor, layer) -> None:
        self._impl = impl
        self._kv_cache = kv_cache
        self._layer = layer

    def write_kept_kv(
        self,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        *,
        block_ids: list[int],
        block_size: int,
    ) -> int:
        return write_kept_kv_chunked(
            self._impl,
            self._kv_cache,
            self._layer,
            k_out,
            v_out,
            block_ids=block_ids,
            block_size=block_size,
        )


def kvarn_kept_kv_writer(
    impl,
    kv_cache: torch.Tensor,
    layer: torch.nn.Module | None = None,
):
    """Return a KeptKvWriter closed over this layer's KVarN impl + cache."""
    return _KvarnKeptKvWriter(impl, kv_cache, layer)
