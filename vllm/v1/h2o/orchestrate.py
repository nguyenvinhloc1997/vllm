# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/orchestrate.py
"""Shared after-attention H2O prefill/decode hooks (backend-agnostic)."""

from __future__ import annotations

import torch

from vllm import envs
from vllm.v1.h2o.bi_snap import bi_snap_budget, select_bi_snap
from vllm.v1.h2o.context import get_h2o_batch_context
from vllm.v1.h2o.runtime import (
    drop_prefill_q,
    get_h2o_runtime,
    get_or_create_prefill_q,
    set_h2o_layer_runtime,
    stash_decode_queries,
)
from vllm.v1.h2o.scores import (
    KvGather,
    accumulate_attention_mass_chunked,
    mass_dict,
)
from vllm.v1.h2o.slots import build_slot_layout
from vllm.v1.h2o.writers import KeptKvWriter


def run_h2o_after_attention(
    *,
    layer_name: str,
    query: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    gather: KvGather,
    writer: KeptKvWriter,
    scale: float,
    sliding_window: tuple[int, int] | None = None,
) -> None:
    """Stash observation Q / gather-score-pack at end of prefill / stash decode Q.

    Call from FA or KVarN after the layer attention forward when ``VLLM_H2O``.
    ``gather`` materializes full K/V once; ``writer`` packs selected K/V.
    """
    if not envs.VLLM_H2O:
        return
    if sliding_window is not None and (
        sliding_window[0] >= 0 or sliding_window[1] >= 0
    ):
        return

    ctx = get_h2o_batch_context()
    if ctx is None or not ctx.requests or ctx.positions is None:
        return

    for request_index, req in enumerate(ctx.requests):
        q = query[req.token_start : req.token_end]
        pos = ctx.positions[req.token_start : req.token_end].tolist()

        if req.num_computed_tokens >= req.prompt_len:
            rt = get_h2o_runtime(req.request_id)
            if rt is not None and layer_name in rt.layers:
                stash_decode_queries(
                    req.request_id,
                    layer_name=layer_name,
                    q=q,
                    positions=pos,
                )
            continue

        stash = get_or_create_prefill_q(
            req.request_id,
            layer_name=layer_name,
            prompt_len=req.prompt_len,
        )
        slots = slot_mapping[req.token_start : req.token_end]
        stash.observe(
            q,
            positions=pos,
            slots=slots.tolist(),
            block_size=block_size,
        )

        if not req.is_last_prefill_chunk:
            continue

        _, recent_len, _ = bi_snap_budget(
            req.prompt_len, ratio=float(envs.VLLM_H2O_RATIO)
        )
        if recent_len >= req.prompt_len:
            drop_prefill_q(req.request_id, layer_name=layer_name)
            continue

        q_obs, q_positions = stash.observations()
        k_full, v_full = gather.gather_kv(
            request_index=request_index,
            slots=stash.slots,
            seq_len=req.prompt_len,
            block_size=block_size,
        )
        if k_full.shape[0] != req.prompt_len or v_full.shape[0] != req.prompt_len:
            raise ValueError("gathered K/V length does not match prompt_len")
        k_positions = list(range(req.prompt_len))
        mass = accumulate_attention_mass_chunked(
            q_obs,
            k_full,
            scale=scale,
            q_positions=q_positions,
            k_positions=k_positions,
        )
        state = select_bi_snap(
            mass_dict(mass, k_positions),
            req.prompt_len,
            ratio=float(envs.VLLM_H2O_RATIO),
        )
        pos_out = sorted(set(state.heavy) | set(state.recent))
        k_out = k_full[pos_out]
        v_out = v_full[pos_out]

        need_blocks = (len(pos_out) + block_size - 1) // block_size
        block_ids = list(stash.block_ids[:need_blocks])
        if len(block_ids) < need_blocks:
            seen = set(block_ids)
            for s in slots.tolist():
                if s < 0:
                    continue
                bid = int(s) // block_size
                if bid not in seen:
                    seen.add(bid)
                    block_ids.append(bid)
                if len(block_ids) >= need_blocks:
                    break
        block_ids = block_ids[:need_blocks]
        if not block_ids:
            continue
        writer.write_kept_kv(k_out, v_out, block_ids=block_ids, block_size=block_size)
        layout = build_slot_layout(pos_out, block_ids, block_size)
        set_h2o_layer_runtime(
            req.request_id,
            layer_name=layer_name,
            state=state,
            layout=layout,
            prompt_len=req.prompt_len,
        )
