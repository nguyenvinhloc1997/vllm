# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Each compress-FA KV group owns its own H2O retained blocks.

Hybrid KV cache groups overlay one backing buffer from byte 0
(`allocate_kv_cache`): block ``b`` of FA group 0 layer ``j`` is the same bytes
as block ``b`` of FA group 1 layer ``j``. Sharing retained block ids across FA
groups therefore makes paired layers overwrite each other's packed KV.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.h2o.context import (
    H2OBatchContext,
    H2ORequestContext,
    clear_h2o_batch_context,
    set_h2o_batch_context,
)
from vllm.v1.h2o.orchestrate import run_h2o_after_attention
from vllm.v1.h2o.policy import compute_k, select_prefill
from vllm.v1.h2o.runner_hooks import maybe_remap_h2o_decode_slots_gpu
from vllm.v1.h2o.runtime import (
    clear_h2o_runtime,
    get_h2o_runtime,
    set_h2o_layer_runtime,
)
from vllm.v1.h2o.slots import build_slot_layout, remap_decode_write_slot

from .test_protect_fa_h2o_scope import FakeGather, FakeWriter, _fa_manager, _FakeCoord

pytestmark = pytest.mark.cpu_test


def test_compress_fa_groups_get_disjoint_retained_blocks():
    block_size, prompt_len, num_keep = 16, 64, 32
    pool = BlockPool(
        num_gpu_blocks=128, enable_caching=False, hash_block_size=block_size
    )
    free_before = pool.get_num_free_blocks()
    g0 = _fa_manager(h2o_protect=False, pool=pool, group_id=0)
    g1 = _fa_manager(h2o_protect=False, pool=pool, group_id=1)
    req_id = "r-disjoint"
    g0.allocate_new_blocks(req_id, prompt_len, prompt_len)
    g1.allocate_new_blocks(req_id, prompt_len, prompt_len)

    coord = _FakeCoord([g0, g1])
    ids = coord.allocate_h2o_retained(req_id, num_keep)
    assert set(ids) == {0, 1}
    assert len(ids[0]) == len(ids[1]) == 2
    assert not set(ids[0]) & set(ids[1])

    coord.resize_h2o_full_attention(req_id, num_keep)
    b0 = [b.block_id for b in g0.req_to_blocks[req_id]]
    b1 = [b.block_id for b in g1.req_to_blocks[req_id]]
    assert b0 == ids[0] and b1 == ids[1]
    assert all(b.ref_cnt == 1 for b in g0.req_to_blocks[req_id])

    g0.free(req_id)
    g1.free(req_id)
    assert pool.get_num_free_blocks() == free_before


def test_abort_frees_every_group_retained_blocks():
    block_size = 16
    pool = BlockPool(
        num_gpu_blocks=128, enable_caching=False, hash_block_size=block_size
    )
    g0 = _fa_manager(h2o_protect=False, pool=pool, group_id=0)
    g1 = _fa_manager(h2o_protect=False, pool=pool, group_id=1)
    coord = _FakeCoord([g0, g1])
    free_before = pool.get_num_free_blocks()
    coord.allocate_h2o_retained("r-abort", 32)
    coord.abort_h2o_retained("r-abort")
    assert pool.get_num_free_blocks() == free_before


def test_orchestrate_packs_each_layer_into_its_group_blocks(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.2)
    clear_h2o_runtime("r0")
    t, h, d = 300, 2, 4
    q = torch.randn(t, 4, d)
    k = torch.randn(t, h, d)
    v = torch.randn(t, h, d)
    layer_a, layer_b = "layers.3.self_attn.attn", "layers.7.self_attn.attn"
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
                    new_block_ids={0: list(range(64)), 1: list(range(100, 164))},
                )
            ],
            positions=torch.arange(t),
            layer_to_group={layer_a: 0, layer_b: 1},
        )
    )
    writer = FakeWriter()
    for layer in (layer_a, layer_b):
        run_h2o_after_attention(
            layer_name=layer,
            query=q,
            slot_mapping=torch.arange(t),
            block_size=16,
            writer=writer,
            scale=1.0,
            gather=FakeGather(k, v),
        )
    assert len(writer.calls) == 2
    ids_a, ids_b = writer.calls[0][1], writer.calls[1][1]
    assert set(ids_a) <= set(range(64))
    assert set(ids_b) <= set(range(100, 164))
    rt = get_h2o_runtime("r0")
    assert rt.layers[layer_a].group_id == 0
    assert rt.layers[layer_b].group_id == 1
    clear_h2o_batch_context()
    clear_h2o_runtime("r0")


def test_decode_remap_writes_each_fa_group_its_own_slot(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    req_id, prompt_len = "r-remap-groups", 40
    clear_h2o_runtime(req_id)
    state = select_prefill(
        {i: float(prompt_len - i) for i in range(prompt_len)},
        prompt_len,
        compute_k(prompt_len, 0.2),
    )
    positions = sorted(set(state.heavy) | set(state.recent))
    layers = {
        "layers.3.self_attn.attn": (0, [0, 1]),
        "layers.7.self_attn.attn": (1, [5, 6]),
    }
    for name, (gid, blocks) in layers.items():
        set_h2o_layer_runtime(
            req_id,
            layer_name=name,
            state=state,
            layout=build_slot_layout(positions, block_ids=blocks, block_size=8),
            prompt_len=prompt_len,
            group_id=gid,
        )
    rt = get_h2o_runtime(req_id)
    expected = {
        gid: remap_decode_write_slot(rt.layers[name].state, rt.layers[name].layout)
        for name, (gid, _) in layers.items()
    }
    assert expected[0] != expected[1]

    kvarn_kv = torch.zeros(4, 2, 256, dtype=torch.uint8)
    forward_ctx = {name: SimpleNamespace(kv_cache=kvarn_kv) for name in layers}
    groups = [
        SimpleNamespace(layer_names=["layers.3.self_attn.attn"]),
        SimpleNamespace(layer_names=["layers.7.self_attn.attn"]),
        SimpleNamespace(layer_names=["layers.0.linear_attn"]),
    ]
    slot_mappings = torch.full((3, 8), -7, dtype=torch.int64)
    maybe_remap_h2o_decode_slots_gpu(
        req_ids=[req_id],
        num_computed_tokens=[prompt_len],
        query_start_loc_np=np.asarray([0, 8]),
        slot_mappings=slot_mappings,
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        forward_ctx=forward_ctx,
    )
    assert (slot_mappings[0] == expected[0]).all()
    assert (slot_mappings[1] == expected[1]).all()
    assert (slot_mappings[2] == -7).all()
    clear_h2o_runtime(req_id)
