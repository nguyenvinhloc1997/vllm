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


def test_bi_snap_m_zero_decode_rotates_recent():
    """When keep <= R, bi-window sets ``k=M=0``; decode must still rotate."""
    from vllm.v1.h2o.policy import decode_step
    from vllm.v1.h2o.slots import (
        build_slot_layout,
        plan_decode_slot_writes,
        remap_decode_write_slot,
    )

    prompt_len = 512
    mass = {i: float(i % 17) for i in range(prompt_len)}
    state = select_bi_snap(mass, prompt_len, ratio=0.4)
    assert state.k == 0
    assert state.heavy == []
    assert len(state.recent) == 256
    assert state.update_heavy is False

    positions = list(state.recent)
    layout = build_slot_layout(
        positions, block_ids=list(range((len(positions) + 127) // 128)), block_size=128
    )
    assert layout.num_keep == 256
    slot = remap_decode_write_slot(state, layout)
    assert slot == layout.slot_for_pos(state.recent[0])

    plan = plan_decode_slot_writes(state, layout, prompt_len, {})
    assert plan.state_after.recent[-1] == prompt_len
    assert len(plan.state_after.recent) == 256
    assert plan.state_after.heavy == []

    after = decode_step(state, prompt_len, {})
    assert after.recent == [*state.recent[1:], prompt_len]
