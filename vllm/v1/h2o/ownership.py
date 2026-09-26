# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/ownership.py
"""Ownership-A: resize full-attention block tables after H2O prefill compress."""

from __future__ import annotations

from vllm.utils.math_utils import cdiv
from vllm.v1.h2o.policy import compute_k


def num_keep_tokens(prompt_len: int, ratio: float = 0.2) -> int:
    """Tokens retained after prefill compress (``2K``, or all if short)."""
    k = compute_k(prompt_len, ratio)
    if k <= 0:
        return prompt_len
    if prompt_len < 2 * k:
        return prompt_len
    return 2 * k


def num_keep_blocks(prompt_len: int, block_size: int, ratio: float = 0.2) -> int:
    """``ceil(2K / block_size)`` (or full prompt blocks when short)."""
    return cdiv(num_keep_tokens(prompt_len, ratio), block_size)
