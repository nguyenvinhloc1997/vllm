# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""H2O prefix-cache isolation and compress budget regressions (CPU)."""

from __future__ import annotations

import pytest
import torch

from tests.v1.core.test_prefix_caching import (
    make_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from tests.v1.h2o.test_compress import _FakeHybridKVCacheManager
from vllm import envs
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
)
from vllm.v1.h2o.compress import compress_prefill_kv, run_prefill_compress_for_request
from vllm.v1.h2o.ownership import num_keep_blocks, num_keep_tokens
from vllm.v1.h2o.policy import compute_k, select_prefill
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def test_short_prompt_under_two_k_budget_keeps_all():
    """Paper prefill: if prompt_len < 2K (K=compute_k), retain the full prompt."""
    ratio = 0.2
    assert compute_k(1, ratio) == 0
    assert num_keep_tokens(1, ratio) == 1
    # prompt_len == 2K exactly → still retains every token.
    prompt_len = 2
    k = compute_k(prompt_len, ratio)
    assert k == 1 and prompt_len == 2 * k
    assert num_keep_tokens(prompt_len, ratio) == prompt_len
    scores = {i: float(i) for i in range(prompt_len)}
    state = select_prefill(scores, prompt_len, k)
    kept = sorted(set(state.heavy) | set(state.recent))
    assert kept == list(range(prompt_len))


def test_short_prompt_compress_is_identity_on_positions():
    torch.manual_seed(0)
    prompt_len = 2
    ratio = 0.2
    assert num_keep_tokens(prompt_len, ratio) == prompt_len
    t, h, d = prompt_len, 1, 4
    q = torch.randn(t, 2, d)
    key = torch.randn(t, h, d)
    val = torch.randn(t, h, d)
    positions = list(range(t))
    _state, _k_out, _v_out, pos_out = compress_prefill_kv(
        q, key, val, positions, ratio=ratio
    )
    assert pos_out == positions


def test_long_prompt_budget_fa_two_k_mamba_unchanged(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.2)

    block_size = 16
    prompt_len = 2500
    ratio = 0.2
    keep = num_keep_tokens(prompt_len, ratio)
    assert keep == 2 * compute_k(prompt_len, ratio)
    keep_blocks = num_keep_blocks(prompt_len, block_size, ratio)

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
        num_gpu_blocks=512, enable_caching=False, hash_block_size=block_size
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
    req_id = "long"
    fa.allocate_new_blocks(req_id, prompt_len, prompt_len)
    mamba.allocate_new_blocks(req_id, prompt_len, prompt_len)
    mamba_before = len(mamba.req_to_blocks[req_id])
    assert mamba_before == cdiv(prompt_len, block_size)

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
    assert fa.h2o_num_tokens[req_id] == keep
    assert len(mamba.req_to_blocks[req_id]) == mamba_before


def test_prefix_cache_h2o_on_off_isolation(monkeypatch):
    block_size = 16
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, 12),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )
    common_token_ids = [i for i in range(3) for _ in range(block_size)]

    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.2)
    req_h2o = make_request("h2o-0", common_token_ids + [99] * 7, block_size, sha256)
    computed, num_hit, _ = manager.get_computed_blocks(req_h2o)
    assert num_hit == 0
    blocks = manager.allocate_slots(req_h2o, len(req_h2o.prompt_token_ids), 0, computed)
    assert blocks is not None
    manager.free(req_h2o)

    monkeypatch.setattr(envs, "VLLM_H2O", False)
    req_off = make_request("off-0", common_token_ids + [100] * 5, block_size, sha256)
    computed_off, num_hit_off, _ = manager.get_computed_blocks(req_off)
    assert num_hit_off == 0
    assert not computed_off.blocks[0]

    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.2)
    req_h2o_2 = make_request("h2o-1", common_token_ids + [101] * 6, block_size, sha256)
    computed_on, num_hit_on, _ = manager.get_computed_blocks(req_h2o_2)
    assert num_hit_on == 3 * block_size
    assert computed_on.get_block_ids() == ([1, 2, 3],)
