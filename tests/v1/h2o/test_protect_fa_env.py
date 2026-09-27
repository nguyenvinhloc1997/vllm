# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_H2O_PROTECT_FA_LAYERS env parsing and FullAttentionSpec tagging."""

from types import SimpleNamespace

import torch

import vllm.envs as envs
from vllm.envs import environment_variables
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec


class _FakeDecoderBackend:
    @staticmethod
    def is_mla() -> bool:
        return False


def _fa_layer(*, layer_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        attn_type=AttentionType.DECODER,
        layer_name=layer_name,
        kv_cache_dtype="auto",
        kv_cache_torch_dtype=torch.bfloat16,
        head_size=64,
        head_size_v=64,
        num_kv_heads=4,
        sliding_window=None,
        get_attn_backend=lambda: _FakeDecoderBackend,
    )


def _vllm_config() -> SimpleNamespace:
    return SimpleNamespace(cache_config=SimpleNamespace(block_size=16))


def test_protect_env_default_empty(monkeypatch):
    monkeypatch.delenv("VLLM_H2O_PROTECT_FA_LAYERS", raising=False)
    assert environment_variables["VLLM_H2O_PROTECT_FA_LAYERS"]() == frozenset()


def test_protect_env_parses(monkeypatch):
    monkeypatch.setenv("VLLM_H2O_PROTECT_FA_LAYERS", "19,47")
    assert environment_variables["VLLM_H2O_PROTECT_FA_LAYERS"]() == frozenset({19, 47})


def test_protect_env_ignores_empty_tokens(monkeypatch):
    monkeypatch.setenv("VLLM_H2O_PROTECT_FA_LAYERS", "19,, 47 , ")
    assert environment_variables["VLLM_H2O_PROTECT_FA_LAYERS"]() == frozenset({19, 47})


def test_get_kv_cache_spec_tags_protected_layers(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O_PROTECT_FA_LAYERS", frozenset({19, 47}))

    spec19 = Attention.get_kv_cache_spec(
        _fa_layer(layer_name="model.layers.19.self_attn"), _vllm_config()
    )
    spec5 = Attention.get_kv_cache_spec(
        _fa_layer(layer_name="model.layers.5.self_attn"), _vllm_config()
    )

    assert isinstance(spec19, FullAttentionSpec)
    assert spec19.h2o_protect is True
    assert isinstance(spec5, FullAttentionSpec)
    assert spec5.h2o_protect is False


def test_get_kv_cache_spec_protect_off_when_env_empty(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O_PROTECT_FA_LAYERS", frozenset())

    spec = Attention.get_kv_cache_spec(
        _fa_layer(layer_name="model.layers.19.self_attn"), _vllm_config()
    )
    assert isinstance(spec, FullAttentionSpec)
    assert spec.h2o_protect is False


def test_get_kv_cache_spec_protect_off_when_layer_index_missing(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O_PROTECT_FA_LAYERS", frozenset({0}))

    spec = Attention.get_kv_cache_spec(
        _fa_layer(layer_name="no.layer.index.here"), _vllm_config()
    )
    assert isinstance(spec, FullAttentionSpec)
    assert spec.h2o_protect is False
