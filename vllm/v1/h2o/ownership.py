# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/ownership.py
"""Ownership-A: resize full-attention block tables after H2O prefill compress."""

from __future__ import annotations

from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.h2o.bi_snap import bi_snap_budget

# KVarN tile / kernel block size on the live serve (group=128).
H2O_KERNEL_BLOCK_SIZE = 128


def num_keep_tokens(prompt_len: int, ratio: float = 0.4) -> int:
    """Tokens retained by bi-window select (Snap middle plus recent window)."""
    _, recent, middle = bi_snap_budget(prompt_len, ratio=ratio)
    return recent + middle


def num_keep_blocks(prompt_len: int, block_size: int, ratio: float = 0.4) -> int:
    """Page count for the retained bi-window K/V."""
    return cdiv(num_keep_tokens(prompt_len, ratio), block_size)


def expand_manager_blocks_to_kernel(
    manager_block_ids: Sequence[int],
    *,
    manager_block_size: int,
    kernel_block_size: int = H2O_KERNEL_BLOCK_SIZE,
    num_keep_tokens: int | None = None,
) -> list[int]:
    """Expand hybrid manager page ids to KVarN kernel tile ids.

    Hybrid serve promotes FA ``block_size`` to match the mamba page (e.g. 1536).
    The runner splits each manager page into ``manager_bs // kernel_bs`` kernel
    tiles (``BlockTable.map_to_kernel_blocks``): manager id ``M`` →
    ``[M*r, M*r+1, …, M*r+r-1]``. Pack/writers address kernel tiles.
    """
    if manager_block_size <= 0 or kernel_block_size <= 0:
        raise ValueError("block sizes must be positive")
    if manager_block_size % kernel_block_size != 0:
        raise ValueError(
            f"manager_block_size {manager_block_size} must be a multiple of "
            f"kernel_block_size {kernel_block_size}"
        )
    ratio = manager_block_size // kernel_block_size
    if ratio == 1:
        out = list(manager_block_ids)
    else:
        out = [int(mid) * ratio + j for mid in manager_block_ids for j in range(ratio)]
    if num_keep_tokens is not None:
        need = cdiv(num_keep_tokens, kernel_block_size)
        out = out[:need]
    return out
