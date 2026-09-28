# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.core.kv_cache_utils import round_retention_interval


@pytest.mark.parametrize(
    "interval, block, expected",
    [
        (None, 1920, None),  # dense stays dense
        (0, 1920, 0),  # "reachable boundaries only" is its own mode
        (9216, 1536, 9216),  # already a multiple
        (13056, 1920, 13440),  # 6.8 blocks -> 7
        (9216, 1920, 9600),  # 4.8 -> 5
        (13056, 2688, 13440),  # 4.86 -> 5
        (100, 1920, 1920),  # below half a block still keeps 1 block
        (2880, 1920, 3840),  # exact half rounds up
    ],
)
def test_round_retention_interval(interval, block, expected):
    assert round_retention_interval(interval, block) == expected
