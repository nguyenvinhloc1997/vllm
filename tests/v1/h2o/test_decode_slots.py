# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# tests/v1/h2o/test_decode_slots.py
"""CPU tests: decode slot remap + dirty-page rewrite (Task 6)."""

from __future__ import annotations

import torch

from vllm.v1.h2o.decode import (
    apply_committed_decode_step,
    next_decode_write_slot,
    score_committed_decode_delta,
)
from vllm.v1.h2o.policy import H2OState, decode_step, select_prefill
from vllm.v1.h2o.rewrite import apply_decode_plan_rewrites, rewrite_dirty_slots
from vllm.v1.h2o.runtime import (
    H2OLayerRuntime,
    clear_all_h2o_runtimes,
    set_h2o_layer_runtime,
)
from vllm.v1.h2o.slots import (
    apply_slot_layout_after_decode,
    build_slot_layout,
    plan_decode_slot_writes,
    remap_decode_write_slot,
)


def _prefill_at_capacity(k: int = 2, prompt_len: int = 10) -> H2OState:
    scores = {i: float(i) for i in range(prompt_len)}
    state = select_prefill(scores, prompt_len, k)
    assert len(state.heavy) == k and len(state.recent) == k
    return state


def test_build_slot_layout_dense_pack():
    positions = [6, 7, 8, 9]
    layout = build_slot_layout(positions, block_ids=[10, 11], block_size=2)
    assert layout.num_keep == 4
    assert layout.slot_for_pos(6) == 10 * 2 + 0
    assert layout.slot_for_pos(9) == 11 * 2 + 1


def test_remap_decode_write_slot_is_aged_recent():
    state = _prefill_at_capacity()
    positions = sorted(set(state.heavy) | set(state.recent))
    layout = build_slot_layout(positions, block_ids=[0], block_size=16)
    slot = remap_decode_write_slot(state, layout)
    assert slot == layout.slot_for_pos(state.recent[0])


def test_plan_decode_drops_weak_heavy_and_reuses_slot():
    state = H2OState(
        k=2,
        heavy=[1, 3],
        recent=[8, 9],
        scores={1: 10.0, 3: 1.0, 8: 5.0, 9: 5.0},
    )
    positions = [1, 3, 8, 9]
    layout = build_slot_layout(positions, block_ids=[0], block_size=16)
    plan = plan_decode_slot_writes(
        state,
        layout,
        new_pos=10,
        new_scores_delta={1: 0.5, 3: 0.0, 8: 0.0, 9: 0.1, 10: 0.0},
    )
    # Aged 8 promoted; 3 dropped — new token writes into 8's slot after copy.
    assert plan.dropped_pos == 3
    assert plan.promote_copy is not None
    assert plan.new_token_slot == layout.slot_for_pos(8)
    assert plan.state_after.recent == [9, 10]
    assert plan.state_after.heavy == [1, 8]


def test_five_decode_steps_slot_layout_stays_2k():
    state = _prefill_at_capacity(k=3, prompt_len=20)
    positions = sorted(set(state.heavy) | set(state.recent))
    layout = build_slot_layout(positions, block_ids=[0, 1], block_size=4)
    layer_rt = H2OLayerRuntime(state=state, layout=layout)
    pos = 20
    for _ in range(5):
        write_before = next_decode_write_slot(layer_rt)
        assert write_before == layer_rt.layout.slot_for_pos(layer_rt.state.recent[0])
        delta = {
            p: 0.05 for p in (set(layer_rt.state.heavy) | set(layer_rt.state.recent))
        }
        delta[pos] = 0.0
        apply_committed_decode_step(layer_rt, new_pos=pos, new_scores_delta=delta)
        assert len(layer_rt.state.heavy) == 3
        assert len(layer_rt.state.recent) == 3
        assert layer_rt.layout.num_keep == 6
        assert set(layer_rt.layout.pos_to_local) == set(layer_rt.state.heavy) | set(
            layer_rt.state.recent
        )
        pos += 1


def test_rewrite_dirty_slots_bf16():
    block_size = 4
    h, d = 2, 8
    key_cache = torch.zeros(2, block_size, h, d, dtype=torch.bfloat16)
    value_cache = torch.zeros(2, block_size, h, d, dtype=torch.bfloat16)
    k = torch.ones(h, d, dtype=torch.float32)
    v = torch.full((h, d), 2.0, dtype=torch.float32)
    from vllm.v1.h2o.rewrite import DirtySlotWrite

    rewrite_dirty_slots(
        key_cache=key_cache,
        value_cache=value_cache,
        block_size=block_size,
        writes=[DirtySlotWrite(slot=5, key=k, value=v)],  # block 1, offset 1
    )
    assert key_cache[1, 1].eq(1).all()
    assert value_cache[1, 1].eq(2).all()


def test_apply_decode_plan_rewrites_promote_then_overwrite():
    state = H2OState(
        k=2,
        heavy=[1, 3],
        recent=[8, 9],
        scores={1: 10.0, 3: 1.0, 8: 5.0, 9: 5.0},
    )
    positions = [1, 3, 8, 9]
    layout = build_slot_layout(positions, block_ids=[0], block_size=16)
    # Seed cache with identifiable values at each local index.
    key_cache = torch.zeros(1, 16, 1, 4, dtype=torch.bfloat16)
    value_cache = torch.zeros(1, 16, 1, 4, dtype=torch.bfloat16)
    for p, loc in layout.pos_to_local.items():
        key_cache[0, loc, 0, :] = float(p)
        value_cache[0, loc, 0, :] = float(p)

    plan = plan_decode_slot_writes(
        state,
        layout,
        new_pos=10,
        new_scores_delta={1: 0.5, 3: 0.0, 8: 0.0, 9: 0.1, 10: 0.0},
    )
    new_key = torch.full((1, 4), 10.0)
    new_value = torch.full((1, 4), 10.0)
    apply_decode_plan_rewrites(
        key_cache=key_cache,
        value_cache=value_cache,
        layout=layout,
        plan=plan,
        new_key=new_key,
        new_value=new_value,
    )
    # Aged 8 copied into dropped 3's slot; new 10 overwrote 8's slot.
    assert key_cache[0, layout.pos_to_local[3], 0, 0].item() == 8.0
    assert key_cache[0, layout.pos_to_local[8], 0, 0].item() == 10.0
    layout2 = apply_slot_layout_after_decode(layout, plan)
    assert 3 not in layout2.pos_to_local
    assert layout2.pos_to_local[8] == layout.pos_to_local[3]
    assert layout2.pos_to_local[10] == layout.pos_to_local[8]


def test_score_committed_decode_delta_only_retained_keys():
    d = 4
    q = torch.zeros(1, 2, d)
    k = torch.zeros(3, 2, d)
    q[0, :, 0] = 1.0
    k[0, :, 0] = 1.0
    k[1, :, 1] = 1.0
    k[2, :, 0] = 0.5
    delta = score_committed_decode_delta(q, k, [10, 11, 12])
    assert set(delta) == {10, 11, 12}
    assert delta[10] > delta[11]


def test_runtime_register_and_clear():
    clear_all_h2o_runtimes()
    state = _prefill_at_capacity()
    positions = sorted(set(state.heavy) | set(state.recent))
    layout = build_slot_layout(positions, block_ids=[0], block_size=16)
    set_h2o_layer_runtime(
        "req-a", layer_name="l0", state=state, layout=layout, prompt_len=10
    )
    from vllm.v1.h2o.runtime import get_h2o_runtime

    assert get_h2o_runtime("req-a") is not None
    clear_all_h2o_runtimes()
    assert get_h2o_runtime("req-a") is None


def test_decode_step_matches_plan_state_after():
    state = _prefill_at_capacity()
    positions = sorted(set(state.heavy) | set(state.recent))
    layout = build_slot_layout(positions, block_ids=[0], block_size=16)
    delta = {p: 0.0 for p in positions}
    delta[10] = 0.0
    plan = plan_decode_slot_writes(state, layout, 10, delta)
    direct = decode_step(state, 10, delta)
    assert plan.state_after.heavy == direct.heavy
    assert plan.state_after.recent == direct.recent
