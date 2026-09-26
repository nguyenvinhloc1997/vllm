# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# tests/v1/h2o/test_compress.py
import torch

from vllm import envs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys
from vllm.v1.h2o.compress import (
    compress_prefill_kv,
    should_compress_prefill,
    try_compress_prefill_kv,
)
from vllm.v1.request import Request


def test_compress_prefill_kv_gathers_selected_positions():
    t, h, d = 8, 2, 4
    q = torch.randn(t, 4, d)
    k = torch.randn(t, h, d)
    v = torch.randn(t, h, d)
    positions = list(range(100, 108))  # absolute positions, not 0..t-1
    state, k_out, v_out, pos_out = compress_prefill_kv(q, k, v, positions, ratio=0.5)
    assert state.k == 2
    assert k_out.shape[0] == 4
    assert v_out.shape[0] == 4
    assert pos_out == sorted(pos_out)
    assert set(pos_out) == set(state.heavy) | set(state.recent)


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
    assert state.k == 2
    assert k_out.shape[0] == 4
    assert v_out.shape[0] == 4
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
