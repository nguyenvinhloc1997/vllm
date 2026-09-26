# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# tests/v1/h2o/test_scores.py
import pytest
import torch

from vllm.v1.h2o.scores import accumulate_attention_mass, mass_dict


def test_decode_query_mass_prefers_aligned_key():
    # 1 query, 2 KV heads, 2 tokens. Query matches key 0.
    d = 4
    q = torch.zeros(1, 2, d)
    k = torch.zeros(2, 2, d)
    q[0, :, 0] = 1.0
    k[0, :, 0] = 1.0
    k[1, :, 1] = 1.0
    mass = accumulate_attention_mass(q, k, scale=d**-0.5)
    assert mass.shape == (2,)
    assert mass[0] > mass[1]
    assert mass_dict(mass, [10, 11])[10] == pytest.approx(mass[0].item())
