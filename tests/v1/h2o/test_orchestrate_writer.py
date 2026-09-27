# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared H2O orchestrator invokes KeptKvWriter at end of prefill."""

from __future__ import annotations

import pytest
import torch

from vllm import envs
from vllm.v1.h2o.context import (
    H2OBatchContext,
    H2ORequestContext,
    clear_h2o_batch_context,
    set_h2o_batch_context,
)
from vllm.v1.h2o.orchestrate import run_h2o_after_attention
from vllm.v1.h2o.runtime import clear_h2o_runtime, get_h2o_runtime

pytestmark = pytest.mark.cpu_test


class FakeWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, list[int], int]] = []

    def write_kept_kv(self, k_out, v_out, *, block_ids, block_size):
        self.calls.append((int(k_out.shape[0]), list(block_ids), int(block_size)))
        return (int(k_out.shape[0]) + block_size - 1) // block_size


class FakeGather:
    def __init__(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k = k
        self.v = v
        self.calls: list[tuple[int, list[int], int]] = []

    def gather_kv(self, *, request_index, slots, seq_len, block_size):
        self.calls.append((request_index, list(slots), seq_len))
        return self.k[:seq_len], self.v[:seq_len]


def test_end_of_prefill_invokes_writer_and_sets_runtime(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.2)
    clear_h2o_runtime("r0")
    clear_h2o_batch_context()
    t, h, d = 300, 2, 4
    q = torch.randn(t, 4, d)
    k = torch.randn(t, h, d)
    v = torch.randn(t, h, d)
    slots = torch.arange(t)
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
                )
            ],
            positions=torch.arange(t),
        )
    )
    w = FakeWriter()
    gather = FakeGather(k, v)
    run_h2o_after_attention(
        layer_name="layers.3",
        query=q,
        slot_mapping=slots,
        block_size=4,
        gather=gather,
        writer=w,
        scale=d**-0.5,
        sliding_window=(-1, -1),
    )
    assert len(gather.calls) == 1
    assert w.calls, "writer must run at end of prefill"
    rt = get_h2o_runtime("r0")
    assert rt is not None and "layers.3" in rt.layers
    clear_h2o_batch_context()
    clear_h2o_runtime("r0")


def test_chunked_prefill_stashes_windows_and_gathers_only_at_end(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.5)
    request_id = "r-chunked"
    clear_h2o_runtime(request_id)
    prompt_len, chunk, h_q, h_kv, d = 600, 200, 2, 1, 4
    q = torch.randn(prompt_len, h_q, d)
    k = torch.randn(prompt_len, h_kv, d)
    v = torch.randn_like(k)
    gather = FakeGather(k, v)
    writer = FakeWriter()

    for start in range(0, prompt_len, chunk):
        end = start + chunk
        set_h2o_batch_context(
            H2OBatchContext(
                requests=[
                    H2ORequestContext(
                        request_id=request_id,
                        num_computed_tokens=start,
                        prompt_len=prompt_len,
                        is_last_prefill_chunk=end == prompt_len,
                        token_start=0,
                        token_end=chunk,
                    )
                ],
                positions=torch.arange(start, end),
            )
        )
        run_h2o_after_attention(
            layer_name="layers.3",
            query=q[start:end],
            slot_mapping=torch.arange(start, end),
            block_size=20,
            gather=gather,
            writer=writer,
            scale=d**-0.5,
            sliding_window=(-1, -1),
        )
        if end < prompt_len:
            assert gather.calls == []

    assert len(gather.calls) == 1
    assert gather.calls[0][1] == list(range(prompt_len))
    assert writer.calls == [(300, list(range(15)), 20)]
    clear_h2o_batch_context()
    clear_h2o_runtime(request_id)


def test_partial_prefix_hit_drops_incomplete_stash_without_gather(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.5)
    request_id = "r-partial-prefix"
    clear_h2o_runtime(request_id)
    prompt_len, start, h_q, h_kv, d = 600, 200, 2, 1, 4
    gather = FakeGather(
        torch.randn(prompt_len, h_kv, d),
        torch.randn(prompt_len, h_kv, d),
    )
    writer = FakeWriter()
    set_h2o_batch_context(
        H2OBatchContext(
            requests=[
                H2ORequestContext(
                    request_id=request_id,
                    num_computed_tokens=start,
                    prompt_len=prompt_len,
                    is_last_prefill_chunk=True,
                    token_start=0,
                    token_end=prompt_len - start,
                )
            ],
            positions=torch.arange(start, prompt_len),
        )
    )

    run_h2o_after_attention(
        layer_name="layers.3",
        query=torch.randn(prompt_len - start, h_q, d),
        slot_mapping=torch.arange(start, prompt_len),
        block_size=20,
        gather=gather,
        writer=writer,
        scale=d**-0.5,
        sliding_window=(-1, -1),
    )

    assert gather.calls == []
    assert writer.calls == []
    rt = get_h2o_runtime(request_id)
    assert rt is not None and rt.prefill_q == {}
    clear_h2o_batch_context()
    clear_h2o_runtime(request_id)


def test_short_prompt_skips_gather_and_drops_q_stash(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.4)
    request_id = "r-short"
    clear_h2o_runtime(request_id)
    t, d = 128, 4
    gather = FakeGather(torch.randn(t, 1, d), torch.randn(t, 1, d))
    writer = FakeWriter()
    set_h2o_batch_context(
        H2OBatchContext(
            requests=[
                H2ORequestContext(
                    request_id=request_id,
                    num_computed_tokens=0,
                    prompt_len=t,
                    is_last_prefill_chunk=True,
                    token_start=0,
                    token_end=t,
                )
            ],
            positions=torch.arange(t),
        )
    )
    run_h2o_after_attention(
        layer_name="layers.3",
        query=torch.randn(t, 2, d),
        slot_mapping=torch.arange(t),
        block_size=16,
        gather=gather,
        writer=writer,
        scale=d**-0.5,
        sliding_window=(-1, -1),
    )

    assert gather.calls == []
    assert writer.calls == []
    rt = get_h2o_runtime(request_id)
    assert rt is not None and rt.prefill_q == {}
    clear_h2o_batch_context()
    clear_h2o_runtime(request_id)
