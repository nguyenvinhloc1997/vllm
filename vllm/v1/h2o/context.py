# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/context.py
"""Per-forward H2O batch context set by the model runner."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class H2ORequestContext:
    """Per-request end-of-prefill gate inputs for one forward."""

    request_id: str
    num_computed_tokens: int
    prompt_len: int
    is_last_prefill_chunk: bool
    # Token slice into the batch Q/K/V / positions tensors.
    token_start: int
    token_end: int


@dataclass
class H2OBatchContext:
    requests: list[H2ORequestContext] = field(default_factory=list)
    positions: torch.Tensor | None = None  # [num_tokens] absolute RoPE positions


_BATCH_CTX: H2OBatchContext | None = None


def set_h2o_batch_context(ctx: H2OBatchContext | None) -> None:
    global _BATCH_CTX
    _BATCH_CTX = ctx


def get_h2o_batch_context() -> H2OBatchContext | None:
    return _BATCH_CTX


def clear_h2o_batch_context() -> None:
    set_h2o_batch_context(None)
