# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H2O Ownership-A / pack / clamp scope: compress FA only (not h2o_protect)."""

from __future__ import annotations

import pytest
import torch

from vllm import envs
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.h2o.context import (
    H2OBatchContext,
    H2ORequestContext,
    clear_h2o_batch_context,
    get_h2o_batch_context,
    set_h2o_batch_context,
)
from vllm.v1.h2o.orchestrate import run_h2o_after_attention
from vllm.v1.h2o.runtime import clear_h2o_runtime, get_h2o_runtime
from vllm.v1.kv_cache_interface import FullAttentionSpec

pytestmark = pytest.mark.cpu_test


class _FakeCoord:
    """Bind coordinator H2O methods onto a bare manager list."""

    def __init__(self, managers: list[FullAttentionManager]) -> None:
        self.single_type_managers = managers

    _compress_fa_managers = KVCacheCoordinator._compress_fa_managers
    allocate_h2o_retained = KVCacheCoordinator.allocate_h2o_retained
    resize_h2o_full_attention = KVCacheCoordinator.resize_h2o_full_attention
    abort_h2o_retained = KVCacheCoordinator.abort_h2o_retained


def _fa_manager(
    *,
    h2o_protect: bool,
    pool: BlockPool,
    group_id: int,
    block_size: int = 16,
) -> FullAttentionManager:
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
        h2o_protect=h2o_protect,
    )
    return FullAttentionManager(
        spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=group_id,
        scheduler_block_size=block_size,
    )


def test_allocate_h2o_skips_protect_managers():
    block_size = 16
    prompt_len = 64
    num_keep = 32
    pool = BlockPool(
        num_gpu_blocks=128, enable_caching=False, hash_block_size=block_size
    )
    compress = _fa_manager(h2o_protect=False, pool=pool, group_id=0)
    protect = _fa_manager(h2o_protect=True, pool=pool, group_id=1)
    req_id = "r-protect-scope"
    compress.allocate_new_blocks(req_id, prompt_len, prompt_len)
    protect.allocate_new_blocks(req_id, prompt_len, prompt_len)
    protect_blocks_before = list(protect.req_to_blocks[req_id])

    coord = _FakeCoord([compress, protect])
    ids = coord.allocate_h2o_retained(req_id, num_keep)
    assert ids, "compress manager must allocate retained blocks"
    assert req_id in compress.h2o_pending_blocks
    assert req_id not in protect.h2o_pending_blocks

    coord.resize_h2o_full_attention(req_id, num_keep)
    assert (
        len(compress.req_to_blocks[req_id]) == (num_keep + block_size - 1) // block_size
    )
    assert protect.req_to_blocks[req_id] == protect_blocks_before
    assert req_id not in protect.h2o_num_tokens


class FakeWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, list[int], int]] = []

    def write_kept_kv(self, k_out, v_out, *, block_ids, block_size):
        self.calls.append((int(k_out.shape[0]), list(block_ids), int(block_size)))
        return (int(k_out.shape[0]) + block_size - 1) // block_size


class FakeGather:
    def __init__(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k = k
        self.v = v
        self.calls: list[tuple[int, list[int], int]] = []

    def gather_kv(self, *, request_index, slots, seq_len, block_size):
        self.calls.append((request_index, list(slots), seq_len))
        return self.k[:seq_len], self.v[:seq_len]


def test_orchestrate_skips_protect_layer(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.2)
    monkeypatch.setattr(envs, "VLLM_H2O_PROTECT_FA_LAYERS", frozenset({19, 47}))
    clear_h2o_runtime("r0")
    clear_h2o_batch_context()
    t, h, d = 300, 2, 4
    q = torch.randn(t, 4, d)
    k = torch.randn(t, h, d)
    v = torch.randn(t, h, d)
    slots = torch.arange(t)
    set_h2o_batch_context(
        H2OBatchContext(
            requests=[
                H2ORequestContext(
                    request_id="r0",
                    num_computed_tokens=0,
                    prompt_len=t,
                    is_last_prefill_chunk=True,
                    token_start=0,
                    token_end=t,
                    new_block_ids={0: list(range(64))},
                )
            ],
            positions=torch.arange(t),
        )
    )
    w = FakeWriter()
    gather = FakeGather(k, v)
    run_h2o_after_attention(
        layer_name="model.layers.19.self_attn",
        query=q,
        slot_mapping=slots,
        block_size=4,
        gather=gather,
        writer=w,
        scale=d**-0.5,
        sliding_window=(-1, -1),
    )
    assert gather.calls == []
    assert w.calls == [], "protect FA layer must not pack"
    ctx = get_h2o_batch_context()
    assert ctx is not None
    assert "r0" not in ctx.failed_pack_request_ids
    assert "r0" not in ctx.packed_request_ids
    assert get_h2o_runtime("r0") is None
    clear_h2o_batch_context()
    clear_h2o_runtime("r0")
