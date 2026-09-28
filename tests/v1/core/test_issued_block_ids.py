# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.worker.utils import get_issued_block_ids, publish_issued_block_ids


def test_block_pool_tracks_freshly_issued_ids():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    a = pool.get_new_blocks(2)
    b = pool.get_new_blocks(1)
    assert pool.take_issued_block_ids() == [x.block_id for x in a + b]
    assert pool.take_issued_block_ids() == []


def test_publish_issued_block_ids_advances_step():
    step0, _ = get_issued_block_ids()
    publish_issued_block_ids([3, 4])
    step1, ids = get_issued_block_ids()
    assert step1 == step0 + 1 and ids == (3, 4)
    publish_issued_block_ids(None)
    assert get_issued_block_ids() == (step1 + 1, ())
