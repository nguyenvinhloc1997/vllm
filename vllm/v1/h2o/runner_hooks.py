# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/runner_hooks.py
"""Shared H2O hooks for V1 and V2 GPU model runners (flag-gated)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

import vllm.envs as envs


def maybe_set_h2o_batch_context(
    *,
    req_ids: Sequence[str],
    num_computed_tokens: Sequence[int] | np.ndarray,
    prompt_lens: Sequence[int] | np.ndarray,
    num_scheduled_tokens: Sequence[int] | np.ndarray,
    query_start_loc_np: np.ndarray,
    positions: torch.Tensor,
) -> None:
    """Publish end-of-prefill gates for H2O FA compress (flag-gated)."""
    if not envs.VLLM_H2O:
        return
    from vllm.v1.h2o.context import (
        H2OBatchContext,
        H2ORequestContext,
        clear_h2o_batch_context,
        set_h2o_batch_context,
    )

    clear_h2o_batch_context()
    num_reqs = len(req_ids)
    req_ctxs: list[H2ORequestContext] = []
    for i, req_id in enumerate(req_ids[:num_reqs]):
        num_computed = int(num_computed_tokens[i])
        prompt_len = int(prompt_lens[i])
        n_sched = int(num_scheduled_tokens[i])
        is_last = num_computed < prompt_len and num_computed + n_sched >= prompt_len
        token_start = int(query_start_loc_np[i])
        token_end = int(query_start_loc_np[i + 1])
        req_ctxs.append(
            H2ORequestContext(
                request_id=req_id,
                num_computed_tokens=num_computed,
                prompt_len=prompt_len,
                is_last_prefill_chunk=is_last,
                token_start=token_start,
                token_end=token_end,
            )
        )
    pos_t = positions[0] if positions.ndim > 1 else positions
    set_h2o_batch_context(H2OBatchContext(requests=req_ctxs, positions=pos_t.detach()))


def maybe_clear_h2o_batch_context() -> list[str]:
    if not envs.VLLM_H2O:
        return []
    from vllm.v1.h2o.context import (
        clear_h2o_batch_context,
        get_h2o_batch_context,
    )

    ctx = get_h2o_batch_context()
    packed_request_ids = (
        sorted(ctx.packed_request_ids - ctx.failed_pack_request_ids)
        if ctx is not None
        else []
    )
    clear_h2o_batch_context()
    return packed_request_ids


def retained_kv_lens_by_batch_idx(req_ids: Sequence[str]) -> dict[int, int]:
    """Map batch row → retained KV length for compressed H2O requests."""
    if not envs.VLLM_H2O:
        return {}
    from vllm.v1.h2o.runtime import get_h2o_runtime

    retained: dict[int, int] = {}
    for i, req_id in enumerate(req_ids):
        rt = get_h2o_runtime(req_id)
        if rt is None or not rt.layers:
            continue
        layer_rt = next(iter(rt.layers.values()))
        retained[i] = int(layer_rt.layout.num_keep)
    return retained


def maybe_clamp_h2o_seq_lens(
    req_ids: Sequence[str],
    *seq_lens_bufs: Any,
) -> None:
    """Clamp FA seq_lens buffers to retained KV length after Ownership-A."""
    if not envs.VLLM_H2O:
        return
    from vllm.v1.h2o.seq_lens import clamp_h2o_seq_lens_inplace

    retained = retained_kv_lens_by_batch_idx(req_ids)
    if not retained:
        return
    for buf in seq_lens_bufs:
        if buf is None:
            continue
        clamp_h2o_seq_lens_inplace(buf, req_retained=retained)


def _stash_aged_slots_for_request(
    *,
    req_id: str,
    forward_ctx: Mapping[str, Any],
) -> int | None:
    """Stash aged-recent KV; return circular write slot from the first layer.

    FA pages get an aged-slot stash for promote-copy. KVarN (and other non-FA)
    layouts still return the circular write slot for ``slot_mapping`` remap but
    skip FA ``split_fa_kv_cache`` / slot reads.
    """
    from vllm.v1.h2o.decode import next_decode_write_slot
    from vllm.v1.h2o.pages import (
        head_size_from_fa_kv_cache,
        is_fa_paged_kv_cache,
        read_slot_kv,
        split_fa_kv_cache,
    )
    from vllm.v1.h2o.runtime import get_h2o_runtime, stash_aged_slot_kv

    rt = get_h2o_runtime(req_id)
    if rt is None or not rt.layers:
        return None
    layer_rt0 = next(iter(rt.layers.values()))
    write_slot = next_decode_write_slot(layer_rt0)
    for layer_name, layer_rt in rt.layers.items():
        attn = forward_ctx.get(layer_name)
        if attn is None or not hasattr(attn, "kv_cache"):
            continue
        kv = attn.kv_cache
        if not isinstance(kv, torch.Tensor) or kv.numel() == 0:
            continue
        if not is_fa_paged_kv_cache(kv):
            continue
        head_size = head_size_from_fa_kv_cache(kv)
        key_cache, value_cache = split_fa_kv_cache(kv, head_size)
        aged_slot = (
            write_slot if layer_rt is layer_rt0 else next_decode_write_slot(layer_rt)
        )
        aged_k, aged_v = read_slot_kv(
            key_cache,
            value_cache,
            aged_slot,
            block_size=layer_rt.layout.block_size,
        )
        stash_aged_slot_kv(layer_rt, key=aged_k, value=aged_v)
    return write_slot


def iter_h2o_decode_slot_remaps(
    *,
    req_ids: Sequence[str],
    num_computed_tokens: Sequence[int] | np.ndarray,
    query_start_loc_np: np.ndarray,
    forward_ctx: Mapping[str, Any],
):
    """Yield ``(tok_idx, write_slot)`` for every decode token needing remap.

    Prefill rows are skipped. After Ownership-A the absolute
    ``TOKEN_TO_KV_SLOT`` indices walk past the truncated block table, so
    **every** scheduled token in the query span (greedy or DFlash drafts)
    must land on the circular-recent head. Spec drafts share that one slot
    until post-commit advances the window — DFlash width is unchanged.
    """
    if not envs.VLLM_H2O:
        return
    from vllm.v1.h2o.runtime import get_h2o_runtime

    for i, req_id in enumerate(req_ids):
        rt = get_h2o_runtime(req_id)
        if rt is None or not rt.layers:
            continue
        num_computed = int(num_computed_tokens[i])
        if num_computed < rt.prompt_len:
            continue
        write_slot = _stash_aged_slots_for_request(
            req_id=req_id, forward_ctx=forward_ctx
        )
        if write_slot is None:
            continue
        tok0 = int(query_start_loc_np[i])
        tok1 = int(query_start_loc_np[i + 1])
        if tok1 <= tok0:
            continue
        for tok_idx in range(tok0, tok1):
            yield tok_idx, write_slot


def maybe_remap_h2o_decode_slots_gpu(
    *,
    req_ids: Sequence[str],
    num_computed_tokens: Sequence[int] | np.ndarray,
    query_start_loc_np: np.ndarray,
    slot_mappings: torch.Tensor,
    kv_cache_config: Any,
    forward_ctx: Mapping[str, Any],
) -> None:
    """Patch V2 GPU ``slot_mappings`` for H2O circular decode writes.

    Only remaps KV cache groups whose layer names appear in the request's
    H2O runtime (full-attention), leaving Mamba/GDN groups untouched.
    """
    if not envs.VLLM_H2O:
        return
    from vllm.v1.h2o.runtime import get_h2o_runtime

    remaps = list(
        iter_h2o_decode_slot_remaps(
            req_ids=req_ids,
            num_computed_tokens=num_computed_tokens,
            query_start_loc_np=query_start_loc_np,
            forward_ctx=forward_ctx,
        )
    )
    if not remaps:
        return

    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
    # Groups are model-static; any H2O runtime's layer set selects FA groups.
    h2o_layers: set[str] | None = None
    for req_id in req_ids:
        rt = get_h2o_runtime(req_id)
        if rt is not None and rt.layers:
            h2o_layers = set(rt.layers.keys())
            break
    fa_group_indices = []
    for g_idx, group in enumerate(groups):
        if g_idx >= slot_mappings.shape[0]:
            break
        if h2o_layers is not None:
            layer_names = getattr(group, "layer_names", ())
            if not any(name in h2o_layers for name in layer_names):
                continue
        fa_group_indices.append(g_idx)
    if not fa_group_indices:
        return
    for tok_idx, write_slot in remaps:
        for g_idx in fa_group_indices:
            slot_mappings[g_idx, tok_idx] = write_slot


def h2o_decode_after_commit(
    commits: dict[str, list[int]],
    forward_ctx: Mapping[str, Any],
) -> None:
    """Apply paper Algorithm 1 decode_step for committed tokens only."""
    if not envs.VLLM_H2O or not commits:
        return
    from vllm.v1.h2o.decode import (
        apply_committed_decode_step,
        apply_committed_decode_step_with_cache,
    )
    from vllm.v1.h2o.pages import (
        head_size_from_fa_kv_cache,
        is_fa_paged_kv_cache,
        split_fa_kv_cache,
    )
    from vllm.v1.h2o.runtime import get_h2o_runtime

    for req_id, positions in commits.items():
        rt = get_h2o_runtime(req_id)
        if rt is None:
            continue
        for layer_name, layer_rt in rt.layers.items():
            attn = forward_ctx.get(layer_name)
            kv = getattr(attn, "kv_cache", None) if attn is not None else None
            use_fa_cache = (
                isinstance(kv, torch.Tensor)
                and kv.numel() > 0
                and is_fa_paged_kv_cache(kv)
            )
            if use_fa_cache:
                head_size = head_size_from_fa_kv_cache(kv)
                key_cache, value_cache = split_fa_kv_cache(kv, head_size)
                for pos in positions:
                    apply_committed_decode_step_with_cache(
                        layer_rt,
                        new_pos=int(pos),
                        key_cache=key_cache,
                        value_cache=value_cache,
                    )
            else:
                # Policy-only: KVarN (and other non-FA) layouts skip FA rewrite.
                for pos in positions:
                    apply_committed_decode_step(
                        layer_rt,
                        new_pos=int(pos),
                        new_scores_delta={},
                    )
            layer_rt.pending_decode_q = None
            layer_rt.pending_decode_positions = []


def maybe_clear_h2o_runtime(req_id: str) -> None:
    if not envs.VLLM_H2O:
        return
    from vllm.v1.h2o.runtime import clear_h2o_runtime

    clear_h2o_runtime(req_id)
