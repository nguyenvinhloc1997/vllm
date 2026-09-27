# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# tests/v1/h2o/test_bi_snap_select.py
"""Bi-window Snap + recent select (Task 13)."""

from vllm.v1.h2o.bi_snap import bi_snap_budget, select_bi_snap


def test_bi_snap_budget_default_ratio():
    keep, r, m = bi_snap_budget(10_000, ratio=0.4, w=256)
    assert keep == 4000
    assert r == 2000
    assert m == 2000


def test_bi_snap_recent_always_kept():
    prompt_len = 100
    # Mass only on early middle; recent must still appear.
    mass = {i: 100.0 if i < 20 else 0.0 for i in range(prompt_len)}
    state = select_bi_snap(mass, prompt_len, ratio=0.4, w=8)
    _, r, _ = bi_snap_budget(prompt_len, ratio=0.4, w=8)
    assert state.recent == list(range(prompt_len - r, prompt_len))
    for p in state.recent:
        assert p in state.scores


def test_bi_snap_front_and_end_mass_both_in_keepers():
    """High mass at front (W0 voters) and near end-of-middle (W voters)."""
    prompt_len = 100
    keep, r, m = bi_snap_budget(prompt_len, ratio=0.4, w=8)
    assert m >= 4
    mass = {i: 0.0 for i in range(prompt_len)}
    # Front cluster
    mass[2] = 50.0
    mass[3] = 49.0
    # End-of-middle cluster (just before recent)
    mass[prompt_len - r - 2] = 48.0
    mass[prompt_len - r - 1] = 47.0
    # Stale middle
    mass[40] = 1.0

    # Pin M so only the four high-mass middle tokens win (budget m=20 would
    # also admit stale index 40 at mass 1.0 ahead of zero-mass ties).
    state = select_bi_snap(mass, prompt_len, ratio=0.4, w=8, middle_budget=4)
    kept = set(state.heavy) | set(state.recent)
    assert 2 in kept and 3 in kept
    assert (prompt_len - r - 2) in kept
    assert (prompt_len - r - 1) in kept
    assert 40 not in state.heavy  # low mass excluded from middle seats


def test_bi_snap_short_prompt_keeps_all():
    prompt_len = 10
    mass = {i: float(i) for i in range(prompt_len)}
    state = select_bi_snap(mass, prompt_len, ratio=0.4, w=256)
    # keep = 4, but w=256 forces R=min(10,256)=10 → keep all in recent
    assert state.heavy == []
    assert state.recent == list(range(prompt_len))
