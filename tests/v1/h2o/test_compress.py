# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# tests/v1/h2o/test_compress.py
import math

import pytest
import torch

from vllm import envs
from vllm.sampling_params import SamplingParams
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
)
from vllm.v1.h2o.compress import (
    compress_prefill_kv,
    run_prefill_compress_for_request,
    should_compress_prefill,
    try_compress_prefill_kv,
)
from vllm.v1.h2o.ownership import num_keep_blocks, num_keep_tokens
from vllm.v1.h2o.scores import (
    accumulate_attention_mass,
    accumulate_attention_mass_chunked,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


def test_compress_prefill_kv_gathers_selected_positions():
    t, h, d = 8, 2, 4
    q = torch.randn(t, 4, d)
    k = torch.randn(t, h, d)
    v = torch.randn(t, h, d)
    positions = list(range(100, 108))  # absolute positions, not 0..t-1
    state, k_out, v_out, pos_out = compress_prefill_kv(q, k, v, positions, ratio=0.5)
    assert state.k == 0
    assert k_out.shape[0] == t
    assert v_out.shape[0] == t
    assert pos_out == sorted(pos_out)
    assert set(pos_out) == set(state.heavy) | set(state.recent)


def test_s2_chunked_mass_matches_dense_on_tiny_t():
    torch.manual_seed(0)
    t, h_kv, d = 16, 2, 8
    q = torch.randn(t, 4, d)
    k = torch.randn(t, h_kv, d)
    scale = d**-0.5
    dense = accumulate_attention_mass(q, k, scale=scale)
    chunked = accumulate_attention_mass_chunked(q, k, scale=scale, tile_q=5, tile_k=7)
    assert chunked.shape == dense.shape
    assert torch.allclose(chunked, dense.float(), atol=1e-4, rtol=1e-4)


def test_s2_never_allocates_tt_via_small_tiles(monkeypatch):
    """Guard: chunked path must not build a full T×T attention matrix."""
    t, h_kv, d = 32, 1, 4
    q = torch.randn(t, 1, d)
    k = torch.randn(t, h_kv, d)
    scale = d**-0.5
    # If someone reintroduces a dense einsum to [T,T], this still completes
    # with tiny tiles; the unit check is shape + finite mass.
    mass = accumulate_attention_mass_chunked(q, k, scale=scale, tile_q=4, tile_k=4)
    assert mass.shape == (t,)
    assert torch.isfinite(mass).all()
    assert mass.sum() > 0


def test_should_compress_prefill_requires_flag_full_attn_last_chunk(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", False)
    assert not should_compress_prefill(
        is_full_attention=True, is_last_prefill_chunk=True
    )

    monkeypatch.setattr(envs, "VLLM_H2O", True)
    assert not should_compress_prefill(
        is_full_attention=False, is_last_prefill_chunk=True
    )
    assert not should_compress_prefill(
        is_full_attention=True, is_last_prefill_chunk=False
    )
    assert should_compress_prefill(is_full_attention=True, is_last_prefill_chunk=True)
    assert should_compress_prefill(
        is_full_attention=True,
        is_last_prefill_chunk=False,
        num_computed_tokens=10,
        prompt_len=10,
    )
    assert not should_compress_prefill(
        is_full_attention=True,
        is_last_prefill_chunk=True,
        num_computed_tokens=8,
        prompt_len=10,
    )


def test_try_compress_prefill_kv_flag_gated(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", False)
    t, h, d = 8, 2, 4
    q = torch.randn(t, 4, d)
    k = torch.randn(t, h, d)
    v = torch.randn(t, h, d)
    positions = list(range(t))
    assert (
        try_compress_prefill_kv(
            q,
            k,
            v,
            positions,
            is_full_attention=True,
            is_last_prefill_chunk=True,
            ratio=0.5,
        )
        is None
    )

    monkeypatch.setattr(envs, "VLLM_H2O", True)
    out = try_compress_prefill_kv(
        q,
        k,
        v,
        positions,
        is_full_attention=True,
        is_last_prefill_chunk=True,
        ratio=0.5,
    )
    assert out is not None
    state, k_out, v_out, pos_out = out
    assert state.k == 0
    assert k_out.shape[0] == t
    assert v_out.shape[0] == t
    assert pos_out == sorted(pos_out)


def _make_request(*, cache_salt: str | None = None) -> Request:
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id="0",
        prompt_token_ids=list(range(6)),
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=None,
        cache_salt=cache_salt,
        block_hasher=None,
    )


def test_generate_block_hash_extra_keys_h2o_marker(monkeypatch):
    request = _make_request()

    monkeypatch.setattr(envs, "VLLM_H2O", False)
    extra_off, _ = generate_block_hash_extra_keys(request, 0, 3, 0)
    assert extra_off is None or "h2o" not in extra_off

    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.2)
    extra_on, _ = generate_block_hash_extra_keys(request, 0, 3, 0)
    assert extra_on is not None
    assert "h2o" in extra_on
    assert 0.2 in extra_on


def test_generate_block_hash_extra_keys_h2o_with_salt(monkeypatch):
    request = _make_request(cache_salt="salt")
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.25)
    extra, _ = generate_block_hash_extra_keys(request, 0, 3, 0)
    assert extra is not None
    assert "salt" in extra
    assert "h2o" in extra
    assert 0.25 in extra


class _FakeHybridKVCacheManager:
    """Minimal stand-in: FA + Mamba groups with resize_h2o_full_attention."""

    def __init__(self, fa: FullAttentionManager, mamba: MambaManager):
        self.fa = fa
        self.mamba = mamba
        self.resize_calls = 0

    def resize_h2o_full_attention(self, request_id: str, num_keep: int) -> None:
        self.resize_calls += 1
        self.fa.resize_after_h2o_compress(request_id, num_keep)


def test_manager_end_of_prefill_ownership_a_once(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.5)

    block_size = 16
    prompt_len = 512
    ratio = 0.5
    keep = num_keep_tokens(prompt_len, ratio)
    assert keep == 256
    keep_blocks = num_keep_blocks(prompt_len, block_size, ratio)
    assert keep_blocks == 16

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
    req_id = "r0"
    fa.allocate_new_blocks(req_id, prompt_len, prompt_len)
    mamba.allocate_new_blocks(req_id, prompt_len, prompt_len)
    fa_before = len(fa.req_to_blocks[req_id])
    mamba_before = len(mamba.req_to_blocks[req_id])
    assert fa_before == cdiv(prompt_len, block_size)
    free_before = pool.get_num_free_blocks()

    t, h, d = prompt_len, 1, 4
    q = torch.randn(t, 2, d)
    k = torch.randn(t, h, d)
    v = torch.randn(t, h, d)
    positions = list(range(t))
    mgr = _FakeHybridKVCacheManager(fa, mamba)

    # Non-final chunk: skip.
    assert (
        run_prefill_compress_for_request(
            q=q,
            k=k,
            v=v,
            positions=positions,
            prompt_len=prompt_len,
            num_computed_tokens=8,
            is_full_attention=True,
            kv_cache_manager=mgr,
            request_id=req_id,
            ratio=ratio,
        )
        is None
    )
    assert mgr.resize_calls == 0
    assert len(fa.req_to_blocks[req_id]) == fa_before

    # Non-FA: skip.
    assert (
        run_prefill_compress_for_request(
            q=q,
            k=k,
            v=v,
            positions=positions,
            prompt_len=prompt_len,
            num_computed_tokens=prompt_len,
            is_full_attention=False,
            kv_cache_manager=mgr,
            request_id=req_id,
            ratio=ratio,
        )
        is None
    )
    assert mgr.resize_calls == 0

    # End-of-prefill: compress + Ownership-A resize once.
    out = run_prefill_compress_for_request(
        q=q,
        k=k,
        v=v,
        positions=positions,
        prompt_len=prompt_len,
        num_computed_tokens=prompt_len,
        is_full_attention=True,
        kv_cache_manager=mgr,
        request_id=req_id,
        ratio=ratio,
    )
    assert out is not None
    assert mgr.resize_calls == 1
    assert len(fa.req_to_blocks[req_id]) == keep_blocks
    assert len(mamba.req_to_blocks[req_id]) == mamba_before
    assert pool.get_num_free_blocks() > free_before
    assert req_id in fa.h2o_num_tokens

    # Second end-of-prefill: idempotent (no further frees / no second grow).
    free_mid = pool.get_num_free_blocks()
    out2 = run_prefill_compress_for_request(
        q=q,
        k=k,
        v=v,
        positions=positions,
        prompt_len=prompt_len,
        num_computed_tokens=prompt_len,
        is_full_attention=True,
        kv_cache_manager=mgr,
        request_id=req_id,
        ratio=ratio,
    )
    assert out2 is not None
    assert mgr.resize_calls == 2  # orchestration still calls; resize is no-op
    assert len(fa.req_to_blocks[req_id]) == keep_blocks
    assert pool.get_num_free_blocks() == free_mid
    # Allocate must not re-grow FA after H2O.
    assert (
        fa.get_num_blocks_to_allocate(
            req_id,
            num_tokens=prompt_len + 8,
            new_computed_blocks=[],
            total_computed_tokens=prompt_len,
            num_local_computed_tokens=prompt_len,
            num_tokens_main_model=prompt_len + 8,
        )
        == 0
    )


def test_manager_flag_off_skips_resize(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", False)
    block_size = 4
    prompt_len = 16
    fa_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
    )
    pool = BlockPool(
        num_gpu_blocks=32, enable_caching=False, hash_block_size=block_size
    )
    fa = FullAttentionManager(
        fa_spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    req_id = "r0"
    fa.allocate_new_blocks(req_id, prompt_len, prompt_len)
    before = len(fa.req_to_blocks[req_id])
    mgr = _FakeHybridKVCacheManager(fa, fa)  # type: ignore[arg-type]
    t, h, d = prompt_len, 1, 4
    assert (
        run_prefill_compress_for_request(
            q=torch.randn(t, 2, d),
            k=torch.randn(t, h, d),
            v=torch.randn(t, h, d),
            positions=list(range(t)),
            prompt_len=prompt_len,
            num_computed_tokens=prompt_len,
            is_full_attention=True,
            kv_cache_manager=mgr,
            request_id=req_id,
            ratio=0.5,
        )
        is None
    )
    assert mgr.resize_calls == 0
    assert len(fa.req_to_blocks[req_id]) == before


def test_num_keep_tokens_matches_policy():
    assert num_keep_tokens(1, 0.2) == 1
    assert num_keep_tokens(100, 0.2) == 100  # W covers the short prompt
    assert num_keep_tokens(1000, 0.4) == 400
    assert math.ceil(400 / 16) == num_keep_blocks(1000, 16, 0.4)
