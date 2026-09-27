# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Task 6b: decode invariants, hybrid-group isolation, GPU graph/spec smoke."""

from __future__ import annotations

import os

import pytest
import torch

from tests.v1.h2o.test_compress import _FakeHybridKVCacheManager
from vllm import envs
from vllm.platforms import current_platform
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
)
from vllm.v1.h2o.compress import run_prefill_compress_for_request
from vllm.v1.h2o.decode import apply_committed_decode_step
from vllm.v1.h2o.ownership import num_keep_blocks, num_keep_tokens
from vllm.v1.h2o.policy import compute_k, decode_step, select_prefill
from vllm.v1.h2o.runtime import H2OLayerRuntime
from vllm.v1.h2o.slots import build_slot_layout
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

pytestmark_cpu = pytest.mark.cpu_test


def _prefill_state(prompt_len: int, ratio: float = 0.2):
    k = compute_k(prompt_len, ratio)
    scores = {i: float(prompt_len - i) for i in range(prompt_len)}
    state = select_prefill(scores, prompt_len, k)
    return state, k


def _delta_for_state(state, new_pos: int) -> dict[int, float]:
    delta = {p: 0.05 for p in (set(state.heavy) | set(state.recent))}
    delta[new_pos] = 0.0
    return delta


@pytestmark_cpu
def test_decode_invariants_budget_dropped_never_return_scores_only_retained():
    """N decode steps: 2K budget; dropped positions gone; scores on retained only."""
    prompt_len = 64
    ratio = 0.2
    state, k = _prefill_state(prompt_len, ratio)
    assert k == 6
    budget = 2 * k
    ever_dropped: set[int] = set()

    pos = prompt_len
    for step in range(12):
        retained = set(state.heavy) | set(state.recent)
        assert len(retained) == budget, f"step {step}"
        assert set(state.scores) == retained
        assert ever_dropped.isdisjoint(retained)

        before = set(state.heavy) | set(state.recent)
        state = decode_step(
            state, new_pos=pos, new_scores_delta=_delta_for_state(state, pos)
        )
        after = set(state.heavy) | set(state.recent)
        ever_dropped |= before - after
        assert state.recent[-1] == pos
        pos += 1

    assert len(ever_dropped) > 0


@pytestmark_cpu
def test_decode_invariants_layer_runtime_matches_policy():
    prompt_len = 40
    state, k = _prefill_state(prompt_len)
    positions = sorted(set(state.heavy) | set(state.recent))
    layout = build_slot_layout(positions, block_ids=[0, 1], block_size=8)
    layer_rt = H2OLayerRuntime(state=state, layout=layout)
    pos = prompt_len
    dropped: set[int] = set()
    for _ in range(8):
        before = set(layer_rt.state.heavy) | set(layer_rt.state.recent)
        plan = apply_committed_decode_step(
            layer_rt,
            new_pos=pos,
            new_scores_delta=_delta_for_state(layer_rt.state, pos),
        )
        after = set(layer_rt.state.heavy) | set(layer_rt.state.recent)
        dropped |= before - after
        assert plan.state_after.heavy == layer_rt.state.heavy
        assert plan.state_after.recent == layer_rt.state.recent
        assert len(after) == 2 * k
        assert set(layer_rt.state.scores) == after
        assert dropped.isdisjoint(after)
        pos += 1


@pytestmark_cpu
def test_hybrid_prefill_compress_and_decode_leaves_mamba_blocks_unchanged(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.8)

    block_size = 16
    prompt_len = 400
    ratio = 0.8
    keep_blocks = num_keep_blocks(prompt_len, block_size, ratio)
    keep = num_keep_tokens(prompt_len, ratio)

    fa_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
    )
    mamba_spec = MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=0,
    )
    pool = BlockPool(
        num_gpu_blocks=128, enable_caching=False, hash_block_size=block_size
    )
    fa = FullAttentionManager(
        fa_spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    mamba = MambaManager(
        mamba_spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=1,
        scheduler_block_size=block_size,
    )
    req_id = "hybrid-decode"
    fa.allocate_new_blocks(req_id, prompt_len, prompt_len)
    mamba.allocate_new_blocks(req_id, prompt_len, prompt_len)
    mamba_before = len(mamba.req_to_blocks[req_id])

    t, h, d = prompt_len, 1, 4
    mgr = _FakeHybridKVCacheManager(fa, mamba)
    out = run_prefill_compress_for_request(
        q=torch.randn(t, 2, d),
        k=torch.randn(t, h, d),
        v=torch.randn(t, h, d),
        positions=list(range(t)),
        prompt_len=prompt_len,
        num_computed_tokens=prompt_len,
        is_full_attention=True,
        kv_cache_manager=mgr,
        request_id=req_id,
        ratio=ratio,
    )
    assert out is not None
    state, _k_out, _v_out, pos_out = out
    assert len(pos_out) == keep
    assert len(fa.req_to_blocks[req_id]) == keep_blocks
    assert len(mamba.req_to_blocks[req_id]) == mamba_before
    assert mgr.resize_calls == 1
    free_after_compress = pool.get_num_free_blocks()

    positions = sorted(pos_out)
    fa_block_ids = [b.block_id for b in fa.req_to_blocks[req_id]]
    layout = build_slot_layout(positions, block_ids=fa_block_ids, block_size=block_size)
    layer_rt = H2OLayerRuntime(state=state, layout=layout)
    pos = prompt_len
    for _ in range(6):
        apply_committed_decode_step(
            layer_rt,
            new_pos=pos,
            new_scores_delta=_delta_for_state(layer_rt.state, pos),
        )
        pos += 1
        assert len(mamba.req_to_blocks[req_id]) == mamba_before
        assert len(fa.req_to_blocks[req_id]) == keep_blocks
        assert mgr.resize_calls == 1

    assert pool.get_num_free_blocks() == free_after_compress


def _cuda_has_free_gib(min_gib: float) -> bool:
    if not current_platform.is_cuda():
        return False
    free_bytes, _total = current_platform.mem_get_info()
    return free_bytes >= int(min_gib * (1024**3))


def _gpu_regression_allowed(*, graph: bool = False) -> bool:
    if not current_platform.is_cuda():
        return False
    if not _cuda_has_free_gib(2.5 if graph else 2.0):
        return False
    return not graph or os.environ.get("VLLM_TEST_H2O_GPU") == "1"


@pytest.fixture(autouse=True)
def _restore_h2o_env(monkeypatch):
    monkeypatch.delenv("VLLM_H2O", raising=False)
    monkeypatch.setattr(envs, "VLLM_H2O", False)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="H2O GPU regression needs CUDA"
)
def test_cuda_graph_smoke_h2o_off_and_on(monkeypatch):
    if not _gpu_regression_allowed(graph=True):
        pytest.skip("Set VLLM_TEST_H2O_GPU=1 and ensure ~2.5GiB free GPU (serve idle)")

    from vllm import LLM, SamplingParams

    prompt = "Hello"
    params = SamplingParams(max_tokens=8, temperature=0.0)

    for h2o_on in (False, True):
        monkeypatch.setenv("VLLM_H2O", "1" if h2o_on else "0")
        monkeypatch.setattr(envs, "VLLM_H2O", h2o_on)
        if h2o_on:
            monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.2)
        llm = LLM(
            model="facebook/opt-125m",
            max_model_len=256,
            enforce_eager=False,
            gpu_memory_utilization=0.18,
        )
        out = llm.generate([prompt], params)[0].outputs[0].token_ids
        assert len(out) >= 1
        del llm


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="H2O GPU regression needs CUDA"
)
def test_h2o_greedy_spec_off_short_prompt(monkeypatch):
    if not _gpu_regression_allowed(graph=False):
        pytest.skip("CUDA device lacks free memory (serve may own the GPU)")

    from vllm import LLM, SamplingParams

    prompt = "The capital of France is"
    params = SamplingParams(max_tokens=12, temperature=0.0)

    monkeypatch.setenv("VLLM_H2O", "1")
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.2)
    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=512,
        enforce_eager=True,
        gpu_memory_utilization=0.15,
    )
    ids = llm.generate([prompt], params)[0].outputs[0].token_ids
    assert len(ids) >= 1
    del llm


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="H2O GPU regression needs CUDA"
)
def test_h2o_short_prompt_ngram_spec_does_not_crash(monkeypatch):
    """Spec on (ngram): H2O flag on must complete a short greedy generate."""
    if not _gpu_regression_allowed(graph=False):
        pytest.skip("CUDA device lacks free memory (serve may own the GPU)")

    from vllm import LLM, SamplingParams

    prompt = "Hi"
    params = SamplingParams(max_tokens=4, temperature=0.0)

    monkeypatch.setenv("VLLM_H2O", "1")
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.2)
    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.15,
        spec_method="ngram",
        spec_tokens=2,
    )
    out = llm.generate([prompt], params)[0].outputs[0].text
    assert isinstance(out, str)
    del llm
