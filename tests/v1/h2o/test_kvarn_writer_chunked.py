# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for chunked KVarN H2O writer pool reclaim."""

from __future__ import annotations

import torch

from vllm.v1.h2o.kvarn_writer import write_kept_kv_chunked


class _FakeKvarnImpl:
    _block_to_slot_dict: dict = {}
    _free_slots: dict = {}
    _block_to_slot_t_per_device: dict = {}
    _max_known_block_id: dict = {}
    _peak_allocated: int = 0
    _flush_bids: list[int]

    def __init__(self, pool_size: int = 2, device: torch.device | None = None):
        self._group_key = ("fake",)
        self.device = device or torch.device("cpu")
        cls = type(self)
        gk = self._group_key
        cls._block_to_slot_dict[gk] = {}
        cls._free_slots[gk] = list(range(pool_size))
        cls._block_to_slot_t_per_device[(self.device, gk)] = torch.full(
            (4096,), -1, dtype=torch.int32
        )
        cls._max_known_block_id[gk] = 0
        cls._peak_allocated = 0
        self._flush_bids = []
        self._update_calls = 0

    def _ensure_pool(self, device, num_blocks_hint: int = 0) -> None:
        return

    def do_kv_cache_update(self, **kwargs) -> None:
        self._update_calls += 1
        gk = self._group_key
        n = len(type(self)._block_to_slot_dict[gk])
        type(self)._peak_allocated = max(type(self)._peak_allocated, n)

    @classmethod
    def _batched_flush(cls, flush_pairs: list) -> None:
        for impl, bid, _kvc in flush_pairs:
            impl._flush_bids.append(int(bid))


def test_chunked_write_reclaims_each_full_tile():
    impl = _FakeKvarnImpl(pool_size=2)
    block_size = 128
    n_tiles = 3
    t_keep = n_tiles * block_size
    k = torch.zeros(t_keep, 4, 8, dtype=torch.float16)
    v = torch.zeros_like(k)
    kv_cache = torch.zeros(1)
    bids = [100, 101, 102]

    n = write_kept_kv_chunked(
        impl,
        kv_cache,
        layer=None,
        k_out=k,
        v_out=v,
        block_ids=bids,
        block_size=block_size,
    )

    assert n == 3
    assert type(impl)._peak_allocated <= 1
    assert impl._flush_bids == [100, 101, 102]
    assert type(impl)._block_to_slot_dict[impl._group_key] == {}
    assert len(type(impl)._free_slots[impl._group_key]) == 2


def test_partial_tail_stays_in_pool():
    impl = _FakeKvarnImpl(pool_size=2)
    block_size = 128
    t_keep = block_size + 10
    k = torch.zeros(t_keep, 4, 8, dtype=torch.float16)
    v = torch.zeros_like(k)
    kv_cache = torch.zeros(1)
    bids = [200, 201]

    write_kept_kv_chunked(
        impl,
        kv_cache,
        layer=None,
        k_out=k,
        v_out=v,
        block_ids=bids,
        block_size=block_size,
    )

    assert impl._flush_bids == [200]
    # Partial tail still mapped.
    assert 201 in type(impl)._block_to_slot_dict[impl._group_key]
    assert 200 not in type(impl)._block_to_slot_dict[impl._group_key]
