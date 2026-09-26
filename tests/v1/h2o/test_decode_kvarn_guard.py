# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Task 10b: decode rewrite skips non-FA KV layouts (KVarN)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm import envs
from vllm.v1.h2o.policy import compute_k, select_prefill
from vllm.v1.h2o.runner_hooks import h2o_decode_after_commit
from vllm.v1.h2o.runtime import (
    H2OLayerRuntime,
    clear_h2o_runtime,
    set_h2o_layer_runtime,
)
from vllm.v1.h2o.slots import build_slot_layout

pytestmark = pytest.mark.cpu_test


def _seed_runtime(req_id: str, layer_name: str, prompt_len: int = 40):
    clear_h2o_runtime(req_id)
    k = compute_k(prompt_len, 0.2)
    scores = {i: float(prompt_len - i) for i in range(prompt_len)}
    state = select_prefill(scores, prompt_len, k)
    positions = sorted(set(state.heavy) | set(state.recent))
    layout = build_slot_layout(positions, block_ids=[0, 1], block_size=8)
    set_h2o_layer_runtime(
        req_id,
        layer_name=layer_name,
        state=state,
        layout=layout,
        prompt_len=prompt_len,
    )
    return get_layer_rt(req_id, layer_name)


def get_layer_rt(req_id: str, layer_name: str) -> H2OLayerRuntime:
    from vllm.v1.h2o.runtime import get_h2o_runtime

    rt = get_h2o_runtime(req_id)
    assert rt is not None
    return rt.layers[layer_name]


def test_decode_skips_fa_split_on_kvarn_layout(monkeypatch):
    """Non-FA (KVarN-shaped) kv_cache must not call split_fa_kv_cache."""
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    req_id = "r-kvarn-guard"
    layer_name = "layers.3.self_attn.attn"
    _seed_runtime(req_id, layer_name)

    # KVarN packed tile: [num_blocks, H_kv, tile_bytes] — not FA 4-D.
    kvarn_kv = torch.zeros(4, 2, 256, dtype=torch.uint8)
    attn = SimpleNamespace(kv_cache=kvarn_kv)
    forward_ctx = {layer_name: attn}

    import vllm.v1.h2o.runner_hooks as hooks

    split_mock = MagicMock(side_effect=AssertionError("must not split KVarN"))
    monkeypatch.setattr(hooks, "split_fa_kv_cache", split_mock, raising=False)
    # Import path uses pages.split inside the function — patch pages.
    import vllm.v1.h2o.pages as pages

    monkeypatch.setattr(
        pages,
        "split_fa_kv_cache",
        MagicMock(side_effect=AssertionError("must not split KVarN")),
    )

    h2o_decode_after_commit({req_id: [40]}, forward_ctx)
    # Policy still advanced without FA rewrite.
    layer_rt = get_layer_rt(req_id, layer_name)
    assert 40 in (set(layer_rt.state.heavy) | set(layer_rt.state.recent))
    clear_h2o_runtime(req_id)


def test_is_fa_paged_kv_cache_detects_layouts():
    from vllm.v1.h2o.pages import is_fa_paged_kv_cache

    fa = torch.zeros(8, 2, 16, 128)  # [B, H, bs, 2D]
    assert is_fa_paged_kv_cache(fa)
    kvarn = torch.zeros(8, 2, 256, dtype=torch.uint8)
    assert not is_fa_paged_kv_cache(kvarn)
