# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/ownership.py
"""Ownership-A: resize full-attention block tables after H2O prefill compress."""

from __future__ import annotations

from vllm.utils.math_utils import cdiv
from vllm.v1.h2o.bi_snap import bi_snap_budget


def num_keep_tokens(prompt_len: int, ratio: float = 0.2) -> int:
    """Tokens retained by bi-window select (Snap middle plus recent window)."""
    _, recent, middle = bi_snap_budget(prompt_len, ratio=ratio)
    return recent + middle


def num_keep_blocks(prompt_len: int, block_size: int, ratio: float = 0.2) -> int:
    """Page count for the retained bi-window K/V."""
    return cdiv(num_keep_tokens(prompt_len, ratio), block_size)
