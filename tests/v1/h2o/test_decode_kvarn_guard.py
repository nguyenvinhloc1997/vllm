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


def test_stash_aged_slots_skips_fa_split_on_kvarn_layout(monkeypatch):
    """Decode slot remap must not FA-split KVarN pages when stashing aged KV."""
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    req_id = "r-kvarn-stash"
    layer_name = "layers.3.self_attn.attn"
    _seed_runtime(req_id, layer_name)
    kvarn_kv = torch.zeros(4, 2, 256, dtype=torch.uint8)
    forward_ctx = {layer_name: SimpleNamespace(kv_cache=kvarn_kv)}

    import vllm.v1.h2o.pages as pages
    from vllm.v1.h2o.runner_hooks import _stash_aged_slots_for_request

    monkeypatch.setattr(
        pages,
        "split_fa_kv_cache",
        MagicMock(side_effect=AssertionError("must not split KVarN")),
    )
    write_slot = _stash_aged_slots_for_request(req_id=req_id, forward_ctx=forward_ctx)
    assert write_slot is not None
    clear_h2o_runtime(req_id)


def test_iter_h2o_decode_slot_remaps_covers_full_query_span(monkeypatch):
    """DFlash schedules N>1 tokens; all must share the circular write slot.

    After Ownership-A the absolute TOKEN_TO_KV_SLOT indices walk past the
    truncated block table. Remapping only tok0 leaves draft slots OOB and
    hangs KVarN decode on the live dflash2 path.
    """
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    req_id = "r-dflash-remap"
    layer_name = "layers.3.self_attn.attn"
    _seed_runtime(req_id, layer_name, prompt_len=40)
    kvarn_kv = torch.zeros(4, 2, 256, dtype=torch.uint8)
    forward_ctx = {layer_name: SimpleNamespace(kv_cache=kvarn_kv)}

    from vllm.v1.h2o.runner_hooks import iter_h2o_decode_slot_remaps
    from vllm.v1.h2o.slots import remap_decode_write_slot

    layer_rt = get_layer_rt(req_id, layer_name)
    expected = remap_decode_write_slot(layer_rt.state, layer_rt.layout)
    # Spec width 8: query span [10, 18)
    remaps = list(
        iter_h2o_decode_slot_remaps(
            req_ids=[req_id],
            num_computed_tokens=[40],
            query_start_loc_np=__import__("numpy").asarray([10, 18]),
            forward_ctx=forward_ctx,
        )
    )
    assert [t for t, _ in remaps] == list(range(10, 18))
    assert all(slots == {0: expected} for _, slots in remaps)
    clear_h2o_runtime(req_id)


def test_is_fa_paged_kv_cache_detects_layouts():
    from vllm.v1.h2o.pages import is_fa_paged_kv_cache

    fa = torch.zeros(8, 2, 16, 128)  # [B, H, bs, 2D]
    assert is_fa_paged_kv_cache(fa)
    kvarn = torch.zeros(8, 2, 256, dtype=torch.uint8)
    assert not is_fa_paged_kv_cache(kvarn)
    # Live KVarN often reinterprets as 4-D [B, group, H_kv, tile_bytes]
    # (tile_bytes even, e.g. 140). Must NOT be treated as FA paged.
    kvarn_4d = torch.zeros(8, 128, 4, 140, dtype=torch.uint8)
    assert not is_fa_paged_kv_cache(kvarn_4d)
    kvarn_4d_fp = torch.zeros(8, 128, 4, 140, dtype=torch.float16)
    assert not is_fa_paged_kv_cache(kvarn_4d_fp)


def test_decode_after_commit_skips_fa_path_on_kvarn_4d(monkeypatch):
    """4-D KVarN reinterpret must stay on policy-only post-commit (no FA score)."""
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    req_id = "r-kvarn-4d"
    layer_name = "layers.3.self_attn.attn"
    _seed_runtime(req_id, layer_name, prompt_len=40)
    # [num_blocks, group, H_kv, tile_bytes] — even last dim, not FA 2*D.
    kvarn_4d = torch.zeros(4, 128, 2, 140, dtype=torch.float16)
    forward_ctx = {layer_name: SimpleNamespace(kv_cache=kvarn_4d)}

    import vllm.v1.h2o.pages as pages

    monkeypatch.setattr(
        pages,
        "split_fa_kv_cache",
        MagicMock(side_effect=AssertionError("must not FA-split KVarN 4-D")),
    )
    h2o_decode_after_commit({req_id: [40]}, forward_ctx)
    layer_rt = get_layer_rt(req_id, layer_name)
    assert 40 in (set(layer_rt.state.heavy) | set(layer_rt.state.recent))
    clear_h2o_runtime(req_id)


def test_bi_window_decode_never_fa_scores_even_on_fa_pages(monkeypatch):
    """Option A: update_heavy=False must not enter Alg-1 FA decode scoring.

    Live crash was h2o_decode_after_commit → FA path → accumulate_attention_mass
    assert on KVarN misdetect. Bi-window freezes heavy; decode is rotate-only.
    """
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    req_id = "r-bi-window-no-fa-score"
    layer_name = "layers.3.self_attn.attn"
    clear_h2o_runtime(req_id)

    from vllm.v1.h2o.bi_snap import select_bi_snap

    prompt_len = 40
    mass = {i: float(40 - i) for i in range(prompt_len)}
    state = select_bi_snap(mass, prompt_len, ratio=0.4, w=8)
    assert state.update_heavy is False
    positions = sorted(set(state.heavy) | set(state.recent))
    layout = build_slot_layout(positions, block_ids=[0, 1, 2, 3], block_size=8)
    set_h2o_layer_runtime(
        req_id,
        layer_name=layer_name,
        state=state,
        layout=layout,
        prompt_len=prompt_len,
    )

    # Classic FA-shaped pages — would previously take the scoring path.
    fa_kv = torch.zeros(4, 2, 8, 256, dtype=torch.float16)  # 2*D=256 → D=128
    forward_ctx = {layer_name: SimpleNamespace(kv_cache=fa_kv)}

    import vllm.v1.h2o.decode as decode_mod
    import vllm.v1.h2o.pages as pages

    monkeypatch.setattr(
        pages,
        "split_fa_kv_cache",
        MagicMock(side_effect=AssertionError("bi-window must not FA-split")),
    )
    monkeypatch.setattr(
        decode_mod,
        "accumulate_attention_mass",
        MagicMock(side_effect=AssertionError("bi-window must not score")),
        raising=False,
    )
    monkeypatch.setattr(
        decode_mod,
        "score_committed_decode_delta",
        MagicMock(side_effect=AssertionError("bi-window must not score")),
    )

    h2o_decode_after_commit({req_id: [40]}, forward_ctx)
    layer_rt = get_layer_rt(req_id, layer_name)
    assert 40 in set(layer_rt.state.recent)
    assert layer_rt.state.update_heavy is False
    clear_h2o_runtime(req_id)
