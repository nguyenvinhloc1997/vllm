# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/writers.py
"""Backend page writers for H2O selected-K/V packing."""

from __future__ import annotations

from typing import Protocol

import torch

from vllm.v1.h2o.compress import repack_kv_into_pages


class KeptKvWriter(Protocol):
    def write_kept_kv(
        self,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        *,
        block_ids: list[int],
        block_size: int,
    ) -> int:
        """Write kept K/V into leading pages. Returns blocks written."""


class _FaKeptKvWriter:
    def __init__(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> None:
        self._key_cache = key_cache
        self._value_cache = value_cache

    def write_kept_kv(
        self,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        *,
        block_ids: list[int],
        block_size: int,
    ) -> int:
        return repack_kv_into_pages(
            k_out,
            v_out,
            key_cache=self._key_cache,
            value_cache=self._value_cache,
            block_ids=block_ids,
            block_size=block_size,
        )


def fa_kept_kv_writer(
    key_cache: torch.Tensor, value_cache: torch.Tensor
) -> KeptKvWriter:
    """FlashAttention paged K/V writer (fp16/bf16 FA layout)."""
    return _FaKeptKvWriter(key_cache, value_cache)
