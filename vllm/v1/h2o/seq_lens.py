# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/seq_lens.py
"""Clamp FA seq_lens to retained KV length after Ownership-A compress."""

from __future__ import annotations


def clamp_h2o_seq_len(absolute_seq_len: int, retained_kv_len: int) -> int:
    """Map transcript length → retained KV length for FA reads.

    After Ownership-A, the physical cache holds ``~2K`` (+ decode growth inside
    the budget). Attention metadata ``seq_lens`` must match that retained
    length; absolute RoPE / transcript length stays on sampling seq_lens
    (``InputBatch.sampling_seq_lens``) so ``seq_len < prefill_len`` is not
    mistaken for chunked prefill.
    """
    if retained_kv_len <= 0:
        return int(absolute_seq_len)
    return int(retained_kv_len)


def clamp_h2o_seq_lens_inplace(
    seq_lens,
    *,
    req_retained: dict[int, int],
) -> None:
    """Clamp per-request entries in a CPU tensor / ndarray (index → retained).

    Args:
        seq_lens: Mutable 1-D buffer (torch CPU tensor or numpy) indexed by
            request row.
        req_retained: ``req_index → retained_kv_len`` for H2O-compressed
            requests only.
    """
    for idx, retained in req_retained.items():
        if idx < 0:
            continue
        try:
            absolute = int(seq_lens[idx])
        except (IndexError, TypeError):
            continue
        seq_lens[idx] = clamp_h2o_seq_len(absolute, retained)
