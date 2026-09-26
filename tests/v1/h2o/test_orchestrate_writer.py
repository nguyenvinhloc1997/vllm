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


def test_end_of_prefill_invokes_writer_and_sets_runtime(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_H2O", True)
    monkeypatch.setattr(envs, "VLLM_H2O_RATIO", 0.2)
    clear_h2o_runtime("r0")
    clear_h2o_batch_context()
    t, h, d = 8, 2, 4
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
    run_h2o_after_attention(
        layer_name="layers.3",
        query=q,
        key=k,
        value=v,
        slot_mapping=slots,
        block_size=4,
        writer=w,
        scale=d**-0.5,
        sliding_window=(-1, -1),
    )
    assert w.calls, "writer must run at end of prefill"
    rt = get_h2o_runtime("r0")
    assert rt is not None and "layers.3" in rt.layers
    clear_h2o_batch_context()
    clear_h2o_runtime("r0")
