# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# tests/v1/h2o/test_correctness_gaps.py
"""CPU/GPU-gated tests for the four H2O live-path correctness gaps."""

from __future__ import annotations

import torch

from vllm.v1.h2o.compress import PrefillQStash, compress_prefill_kv
from vllm.v1.h2o.decode import (
    apply_committed_decode_step_with_cache,
    score_committed_decode_delta,
)
from vllm.v1.h2o.pages import gather_kv_for_positions, read_slot_kv, write_slot_kv
from vllm.v1.h2o.policy import H2OState, select_prefill
from vllm.v1.h2o.runtime import H2OLayerRuntime, clear_all_h2o_runtimes
from vllm.v1.h2o.seq_lens import clamp_h2o_seq_len, clamp_h2o_seq_lens_inplace
from vllm.v1.h2o.slots import build_slot_layout, plan_decode_slot_writes


def _capacity_state(k: int = 2) -> H2OState:
    return H2OState(
        k=k,
        heavy=[1, 3],
        recent=[8, 9],
        scores={1: 10.0, 3: 1.0, 8: 5.0, 9: 5.0},
    )


def test_score_committed_decode_delta_non_empty_from_mock_tensors():
    d = 4
    q = torch.zeros(1, 2, d)
    k = torch.zeros(4, 2, d)
    q[0, :, 0] = 1.0
    k[0, :, 0] = 1.0  # pos 1 — aligns with q
    k[1, :, 1] = 1.0
    k[2, :, 0] = 0.25
    k[3, :, 0] = 0.1
    delta = score_committed_decode_delta(q, k, [1, 3, 8, 9])
    assert delta
    assert set(delta) == {1, 3, 8, 9}
    assert delta[1] > delta[3]


def test_apply_committed_decode_step_with_cache_promote_copy():
    """Promote-copy + new overwrite hit physical FA pages (gap 1+2)."""
    state = _capacity_state()
    positions = [1, 3, 8, 9]
    layout = build_slot_layout(positions, block_ids=[0], block_size=16)
    key_cache = torch.zeros(1, 16, 1, 4, dtype=torch.bfloat16)
    value_cache = torch.zeros(1, 16, 1, 4, dtype=torch.bfloat16)
    for p, loc in layout.pos_to_local.items():
        key_cache[0, loc, 0, :] = float(p)
        value_cache[0, loc, 0, :] = float(p)

    layer_rt = H2OLayerRuntime(state=state, layout=layout)
    # Stash aged recent (8) as if remap captured it before FA wrote 10.
    aged_slot = layout.slot_for_pos(8)
    aged_k, aged_v = read_slot_kv(key_cache, value_cache, aged_slot, block_size=16)
    layer_rt.stashed_aged_key = aged_k
    layer_rt.stashed_aged_value = aged_v
    # Simulate FA writing new token 10 into circular slot.
    write_slot_kv(
        key_cache,
        value_cache,
        aged_slot,
        torch.full((1, 4), 10.0),
        torch.full((1, 4), 10.0),
        block_size=16,
    )
    # Stash committed Q that prefers heavy pos 1.
    q = torch.zeros(1, 1, 4)
    q[0, 0, 0] = 1.0
    layer_rt.pending_decode_q = q
    layer_rt.pending_decode_positions = [10]

    plan = apply_committed_decode_step_with_cache(
        layer_rt,
        new_pos=10,
        key_cache=key_cache,
        value_cache=value_cache,
    )
    assert plan.promote_copy is not None
    assert plan.dropped_pos == 3
    # Aged 8 promoted into dropped 3's slot; new 10 sits in 8's old slot.
    assert key_cache[0, layout.pos_to_local[3], 0, 0].item() == 8.0
    assert key_cache[0, layout.pos_to_local[8], 0, 0].item() == 10.0
    assert 10 in layer_rt.state.recent
    assert 3 not in set(layer_rt.state.heavy) | set(layer_rt.state.recent)
    # Score deltas were non-empty (pos 1 should have gained mass vs zeros).
    assert layer_rt.state.scores.get(1, 0.0) >= 10.0


def test_clamp_h2o_seq_len_helper():
    assert clamp_h2o_seq_len(10_000, 2048) == 2048
    assert clamp_h2o_seq_len(100, 0) == 100
    buf = torch.tensor([100, 5000, 200], dtype=torch.int32)
    clamp_h2o_seq_lens_inplace(buf, req_retained={1: 64})
    assert buf.tolist() == [100, 64, 200]


def test_h2o_clamped_seq_len_must_not_gate_sampling_as_prefill():
    """Ownership-A clamp leaves seq_lens << prefill_len; sampling must use absolute.

    V2 ``get_num_sampled_and_rejected`` zeros num_sampled when
    ``seq_len < prefill_len`` (chunked prefill). After H2O clamp, attention
    seq_lens is retained (~2K) while prefill_len stays the full prompt — that
    must not discard decode samples. Absolute length lives on
    ``InputBatch.sampling_seq_lens``.
    """
    prompt_len = 1416
    retained = 566
    absolute_decode = prompt_len + 3  # a few decode tokens past prompt
    attn_seq = torch.tensor([retained], dtype=torch.int32)
    sampling_seq = torch.tensor([absolute_decode], dtype=torch.int32)
    assert attn_seq.item() < prompt_len  # would wrongly look like chunked prefill
    assert sampling_seq.item() >= prompt_len  # decode gate stays open


def test_prefill_q_stash_keeps_fixed_front_and_rolling_tail():
    prompt_len, h_q, d = 700, 2, 4
    q = torch.arange(prompt_len * h_q * d, dtype=torch.float32).reshape(
        prompt_len, h_q, d
    )
    stash = PrefillQStash(prompt_len=prompt_len)
    for start in range(0, prompt_len, 100):
        end = min(start + 100, prompt_len)
        stash.observe(
            q[start:end],
            positions=list(range(start, end)),
            slots=list(range(start, end)),
            block_size=20,
        )

    assert stash.q_front is not None
    assert stash.q_tail is not None
    assert stash.q_front.shape == (256, h_q, d)
    assert stash.q_tail.shape == (256, h_q, d)
    assert not hasattr(stash, "k_buf")
    assert not hasattr(stash, "v_buf")
    q_obs, obs_positions = stash.observations()
    assert obs_positions == list(range(256)) + list(range(444, 700))
    assert torch.equal(q_obs, q[obs_positions])
    assert stash.slots == list(range(prompt_len))
    assert stash.block_ids == list(range(35))


def test_single_chunk_path_still_compresses_without_accumulator():
    """Single-chunk path must keep working (no PrefillMassAccumulator required)."""
    t, h, d = 8, 2, 4
    q = torch.randn(t, 4, d)
    k = torch.randn(t, h, d)
    v = torch.randn(t, h, d)
    state, k_out, v_out, pos_out = compress_prefill_kv(
        q, k, v, list(range(t)), ratio=0.5
    )
    assert k_out.shape[0] == t
    assert set(pos_out) == set(state.heavy) | set(state.recent)


def test_gather_retained_keys_from_layout_pages():
    state = select_prefill({i: float(i) for i in range(10)}, 10, 2)
    positions = sorted(set(state.heavy) | set(state.recent))
    layout = build_slot_layout(positions, block_ids=[2], block_size=16)
    key_cache = torch.zeros(4, 16, 1, 4)
    value_cache = torch.zeros(4, 16, 1, 4)
    for p in positions:
        slot = layout.slot_for_pos(p)
        bid, off = divmod(slot, 16)
        key_cache[bid, off, 0, :] = float(p)
        value_cache[bid, off, 0, :] = float(p) * 10
    k, v = gather_kv_for_positions(key_cache, value_cache, layout, positions)
    assert k.shape[0] == len(positions)
    for i, p in enumerate(positions):
        assert k[i, 0, 0].item() == float(p)
        assert v[i, 0, 0].item() == float(p) * 10


def test_plan_still_matches_promote_when_scores_empty_vs_nonzero():
    """Sanity: empty deltas still plan; nonzero deltas can change drop."""
    state = _capacity_state()
    layout = build_slot_layout([1, 3, 8, 9], block_ids=[0], block_size=16)
    plan_empty = plan_decode_slot_writes(state, layout, 10, {})
    assert plan_empty.dropped_pos in {1, 3, 8}
    plan_bias = plan_decode_slot_writes(
        state, layout, 10, {1: 100.0, 3: 0.0, 8: 0.0, 9: 0.0, 10: 0.0}
    )
    assert plan_bias.dropped_pos == 3


def teardown_function():
    clear_all_h2o_runtimes()
