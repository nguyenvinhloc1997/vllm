# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# tests/v1/h2o/test_policy.py
import pytest

from vllm.v1.h2o.policy import compute_k, decode_step, select_prefill


def test_compute_k_twenty_percent_even():
    # 61492 * 0.2 = 12298.4 → floor to even total 12298 → K = 6149
    assert compute_k(61492, 0.2) == 6149


def test_compute_k_short_prompt():
    assert compute_k(10, 0.2) == 1  # 20% of 10 = 2 → K = 1


def test_select_prefill_keeps_recent_and_top_heavy():
    prompt_len = 10
    k = 2
    # scores favor positions 1 and 3 in the non-recent prefix [0..7]
    scores = {i: float(i) for i in range(prompt_len)}
    state = select_prefill(scores, prompt_len, k)
    assert state.recent == [8, 9]
    assert state.heavy == [6, 7]  # top scores among 0..7
    assert set(state.heavy) | set(state.recent) == {6, 7, 8, 9}


def test_decode_step_rotates_recent_and_evicts_weak_heavy():
    from vllm.v1.h2o.policy import H2OState

    state = H2OState(
        k=2,
        heavy=[1, 3],
        recent=[8, 9],
        scores={1: 10.0, 3: 1.0, 8: 5.0, 9: 5.0},
    )
    # new token 10; age out 8 (score 5). Among heavy∪{8} = {1,3,8}, drop 3 (lowest).
    out = decode_step(
        state, new_pos=10, new_scores_delta={1: 0.5, 3: 0.0, 8: 0.0, 9: 0.1, 10: 0.0}
    )
    assert out.recent == [9, 10]
    assert out.heavy == [1, 8]
    assert 3 not in out.heavy
    assert out.scores[1] == pytest.approx(10.5)
