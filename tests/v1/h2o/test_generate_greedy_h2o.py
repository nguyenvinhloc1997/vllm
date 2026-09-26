# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional GPU: short-prompt greedy equivalence with H2O on vs off."""

from __future__ import annotations

import pytest

from vllm import envs
from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_cuda(), reason="GPU greedy H2O test needs CUDA"
    ),
]


def _cuda_has_free_gib(min_gib: float) -> bool:
    if not current_platform.is_cuda():
        return False
    free_bytes, _total = current_platform.mem_get_info()
    return free_bytes >= int(min_gib * (1024**3))


@pytest.fixture(autouse=True)
def _restore_h2o_env(monkeypatch):
    monkeypatch.delenv("VLLM_H2O", raising=False)
    monkeypatch.setattr(envs, "VLLM_H2O", False)


def test_short_prompt_greedy_h2o_on_matches_off(monkeypatch):
    if not _cuda_has_free_gib(2.0):
        pytest.skip("CUDA device lacks free memory (serve may own the GPU)")

    from vllm import LLM, SamplingParams

    prompt = "The capital of France is"
    params = SamplingParams(max_tokens=16, temperature=0.0)

    monkeypatch.setenv("VLLM_H2O", "0")
    monkeypatch.setattr(envs, "VLLM_H2O", False)
    llm_off = LLM(
        model="facebook/opt-125m",
        max_model_len=512,
        enforce_eager=True,
        gpu_memory_utilization=0.15,
    )
    off_ids = llm_off.generate([prompt], params)[0].outputs[0].token_ids
    del llm_off

    monkeypatch.setenv("VLLM_H2O", "1")
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.2)
    llm_on = LLM(
        model="facebook/opt-125m",
        max_model_len=512,
        enforce_eager=True,
        gpu_memory_utilization=0.15,
    )
    on_ids = llm_on.generate([prompt], params)[0].outputs[0].token_ids
    del llm_on

    assert on_ids == off_ids
