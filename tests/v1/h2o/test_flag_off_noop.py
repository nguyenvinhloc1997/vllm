# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""H2O linked with VLLM_H2O=0 must not change stock prefix-cache behavior."""

from __future__ import annotations

import importlib

import pytest

# Ensure the H2O package is importable (Task 5 wiring) without toggling the flag.
import vllm.v1.h2o.compress  # noqa: F401
import vllm.v1.h2o.context  # noqa: F401
import vllm.v1.h2o.ownership  # noqa: F401
from tests.v1.core.test_prefix_caching import (
    make_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from vllm import envs
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    generate_block_hash_extra_keys,
    get_request_block_hasher,
    hash_block_tokens,
    init_none_hash,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def _vanilla_request() -> Request:
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id="noop",
        prompt_token_ids=[i % 17 for i in range(55)],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=None,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def test_flag_off_extra_keys_have_no_h2o_marker(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", False)
    request = _vanilla_request()
    extra, _ = generate_block_hash_extra_keys(request, 0, 16, 0)
    assert extra is None or "h2o" not in extra


def test_flag_off_block_hash_chain_matches_stock_recipe(monkeypatch):
    """Import path + flag off → same chained hashes as pre-H2O token recipe."""
    monkeypatch.setattr(envs, "VLLM_H2O", False)
    block_size = 16
    token_ids = [i for i in range(3) for _ in range(block_size)] + [99] * 7

    req = make_request("baseline", token_ids, block_size, sha256)
    assert len(req.block_hashes) == 3

    parent = None
    expected: list = []
    for block_idx in range(3):
        start = block_idx * block_size
        end = start + block_size
        extra, _ = generate_block_hash_extra_keys(req, start, end, 0)
        assert extra is None or "h2o" not in extra
        block_hash = hash_block_tokens(
            sha256, parent, token_ids[start:end], extra_keys=extra
        )
        expected.append(block_hash)
        parent = block_hash

    assert req.block_hashes == expected


def test_h2o_modules_reload_does_not_change_flag_off_hashes(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", False)
    req_before = make_request("a", list(range(48)), 16, sha256)
    hashes_before = list(req_before.block_hashes)

    importlib.reload(vllm.v1.h2o.compress)
    importlib.reload(vllm.v1.core.kv_cache_utils)

    req_after = make_request("b", list(range(48)), 16, sha256)
    assert req_after.block_hashes == hashes_before


def test_flag_off_manager_allocate_free_matches_golden(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", False)
    block_size = 16
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )
    common_token_ids = [i for i in range(3) for _ in range(block_size)]
    unique_token_ids = [3] * 7
    req = make_request("0", common_token_ids + unique_token_ids, block_size, sha256)
    computed_blocks, num_computed_tokens, _ = manager.get_computed_blocks(req)
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req, 55, len(computed_blocks.blocks[0]) * block_size, computed_blocks
    )
    assert blocks is not None and blocks.get_block_ids() == ([1, 2, 3, 4],)
    manager.free(req)
    assert manager.block_pool.free_block_queue.num_free_blocks == 10
