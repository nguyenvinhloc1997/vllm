# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/orchestrate.py
"""Shared after-attention H2O prefill/decode hooks (backend-agnostic)."""

from __future__ import annotations

import torch

from vllm import envs
from vllm.v1.h2o.compress import after_full_attention_kv_update
from vllm.v1.h2o.context import get_h2o_batch_context
from vllm.v1.h2o.ownership import num_keep_tokens
from vllm.v1.h2o.runtime import (
    get_h2o_runtime,
    get_or_create_prefill_mass,
    set_h2o_layer_runtime,
    stash_decode_queries,
)
from vllm.v1.h2o.slots import build_slot_layout
from vllm.v1.h2o.writers import KeptKvWriter


def run_h2o_after_attention(
    *,
    layer_name: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    writer: KeptKvWriter,
    scale: float,
    sliding_window: tuple[int, int] | None = None,
) -> None:
    """Accumulate S2 mass / end-of-prefill pack / stash decode Q.

    Call from FA or KVarN after the layer attention forward when ``VLLM_H2O``.
    ``writer`` packs Algorithm-1 kept K/V into leading pages for that backend.
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

    for req in ctx.requests:
        q = query[req.token_start : req.token_end]
        k = key[req.token_start : req.token_end]
        v = value[req.token_start : req.token_end]
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

        mass_acc = get_or_create_prefill_mass(
            req.request_id,
            layer_name=layer_name,
            prompt_len=req.prompt_len,
        )
        slots = slot_mapping[req.token_start : req.token_end]
        mass_acc.note_slots(slots.tolist(), block_size=block_size)
        mass_acc.update(
            q,
            k,
            q_positions=pos,
            k_positions=pos,
            scale=scale,
            v=v,
        )

        if not req.is_last_prefill_chunk:
            continue

        if q.shape[0] == req.prompt_len:
            out = after_full_attention_kv_update(
                key=k,
                value=v,
                query=q,
                positions=pos,
                is_last_prefill_chunk=True,
                num_computed_tokens=req.prompt_len,
                prompt_len=req.prompt_len,
                sliding_window=sliding_window,
            )
        else:
            full = mass_acc.full_kv()
            if full is None:
                continue
            k_full, v_full, pos_full = full
            out = after_full_attention_kv_update(
                key=k_full,
                value=v_full,
                query=None,
                positions=pos_full,
                is_last_prefill_chunk=True,
                num_computed_tokens=req.prompt_len,
                prompt_len=req.prompt_len,
                sliding_window=sliding_window,
                mass=mass_acc.mass,
            )
        if out is None:
            continue
        state, k_out, v_out, pos_out = out
        keep_n = num_keep_tokens(req.prompt_len, float(envs.VLLM_H2O_RATIO))
        need_blocks = (keep_n + block_size - 1) // block_size if keep_n else 0
        block_ids = list(mass_acc.block_ids[:need_blocks])
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
