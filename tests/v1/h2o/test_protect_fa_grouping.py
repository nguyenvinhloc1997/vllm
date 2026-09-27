# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline H2O protect-FA grouping probe (step 0a).

Naive 14 compress + 2 protect FA buckets collapse hybrid group_size to 2
even with vendor prefer-SWA; peel lands in a later task.
"""

import pytest
import torch

from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_groups_uniform_page_size,
    _probe_group_size_for_test,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec


def _fa(h2o_protect: bool = False) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
        h2o_protect=h2o_protect,
    )


def _sw() -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
        sliding_window=2048,
    )


def _baseline_16fa_5sw() -> dict:
    spec = {f"fa.{i}": _fa(False) for i in range(16)}
    spec.update({f"sw.{i}": _sw() for i in range(5)})
    return spec


def _naive_14c_2p_5sw() -> dict:
    spec = {f"fa_c.{i}": _fa(False) for i in range(14)}
    spec.update({f"fa_p.{i}": _fa(True) for i in range(2)})
    spec.update({f"sw.{i}": _sw() for i in range(5)})
    return spec


def test_baseline_prefer_swa_group_size():
    """16 FA + 5 SWA: prefer-SWA lifts min=5 to group_size 8."""
    assert _probe_group_size_for_test(_baseline_16fa_5sw()) == 8


def test_naive_protect_bucket_collapses_group_size():
    """14 compress + 2 protect FA + 5 SWA without peel: min bucket=2 → group_size 2.

    Assertion uses `_probe_group_size_for_test` (shared bucketing + prefer-SWA
    with `_get_kv_cache_groups_uniform_page_size`), not reverse-engineered
    group counts. See lab note
    hyperqwen/eviction/2026-09-27-h2o-protect-fa-grouping-probe.md.
    """
    spec = _naive_14c_2p_5sw()
    # Smoke: grouping still builds (protect/compress stay separate buckets).
    groups = _get_kv_cache_groups_uniform_page_size(spec)
    assert len(groups) > 0
    assert _probe_group_size_for_test(spec) == 2


def test_full_attention_merge_rejects_mixed_h2o_protect():
    with pytest.raises(ValueError, match="h2o_protect"):
        FullAttentionSpec.merge([_fa(False), _fa(True)])
