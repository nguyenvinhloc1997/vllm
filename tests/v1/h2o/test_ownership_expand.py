# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.h2o.ownership import expand_manager_blocks_to_kernel


def test_expand_manager_blocks_identity_when_equal():
    assert expand_manager_blocks_to_kernel(
        [3, 7], manager_block_size=128, kernel_block_size=128
    ) == [3, 7]


def test_expand_manager_blocks_hybrid_1536():
    # manager 0 → tiles 0..11; manager 1 → 12..23
    out = expand_manager_blocks_to_kernel(
        [0, 1],
        manager_block_size=1536,
        kernel_block_size=128,
    )
    assert out == list(range(24))


def test_expand_manager_blocks_truncates_to_keep():
    out = expand_manager_blocks_to_kernel(
        [5, 6, 7, 8],
        manager_block_size=1536,
        kernel_block_size=128,
        num_keep_tokens=5712,  # need 45 tiles; 4 pages = 48
    )
    assert len(out) == 45
    assert out[0] == 5 * 12
    assert out[-1] == 5 * 12 + 44
