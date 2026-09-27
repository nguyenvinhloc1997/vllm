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
from vllm.logger import init_logger

logger = init_logger(__name__)


def maybe_set_h2o_batch_context(
    *,
    req_ids: Sequence[str],
    num_computed_tokens: Sequence[int] | np.ndarray,
    prompt_lens: Sequence[int] | np.ndarray,
    num_scheduled_tokens: Sequence[int] | np.ndarray,
    query_start_loc_np: np.ndarray,
    positions: torch.Tensor,
    h2o_new_block_ids: Mapping[str, Mapping[int, Sequence[int]]] | None = None,
    kv_cache_config: Any = None,
) -> None:
    """Publish end-of-prefill gates for H2O compress (flag-gated)."""
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
    new_ids_map = h2o_new_block_ids or {}
    req_ctxs: list[H2ORequestContext] = []
    for i, req_id in enumerate(req_ids[:num_reqs]):
        num_computed = int(num_computed_tokens[i])
        prompt_len = int(prompt_lens[i])
        n_sched = int(num_scheduled_tokens[i])
        is_last = num_computed < prompt_len and num_computed + n_sched >= prompt_len
        token_start = int(query_start_loc_np[i])
        token_end = int(query_start_loc_np[i + 1])
        nb = new_ids_map.get(req_id)
        if is_last or nb is not None:
            logger.info(
                "[H2O_DIAG] batch_ctx req=%s prompt=%d computed=%d sched=%d "
                "is_last=%s new_ids=%s map_keys=%s",
                req_id,
                prompt_len,
                num_computed,
                n_sched,
                is_last,
                None if nb is None else {g: len(b) for g, b in nb.items()},
                list(new_ids_map.keys()),
            )
        req_ctxs.append(
            H2ORequestContext(
                request_id=req_id,
                num_computed_tokens=num_computed,
                prompt_len=prompt_len,
                is_last_prefill_chunk=is_last,
                token_start=token_start,
                token_end=token_end,
                new_block_ids=(
                    {int(g): list(b) for g, b in nb.items()} if nb is not None else None
                ),
            )
        )
    pos_t = positions[0] if positions.ndim > 1 else positions
    set_h2o_batch_context(
        H2OBatchContext(
            requests=req_ctxs,
            positions=pos_t.detach(),
            layer_to_group=_layer_to_group(kv_cache_config),
        )
    )


def _layer_to_group(kv_cache_config: Any) -> dict[str, int]:
    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
    return {
        name: gid
        for gid, group in enumerate(groups)
        for name in getattr(group, "layer_names", ())
    }


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
    if ctx is not None:
        logger.info(
            "[H2O_DIAG] clear_batch_ctx packed=%s failed=%s result=%s",
            sorted(ctx.packed_request_ids),
            sorted(ctx.failed_pack_request_ids),
            packed_request_ids,
        )
    else:
        logger.info("[H2O_DIAG] clear_batch_ctx ctx=None")
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
) -> dict[int, int] | None:
    """Stash aged-recent KV; return circular write slot per KV cache group.

    Each compress-FA group owns its retained blocks, so the write slot is taken
    from the first layer of each group (layers in a group share one
    slot_mapping row).

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
    slots: dict[int, int] = {}
    for layer_name, layer_rt in rt.layers.items():
        slot = slots.setdefault(layer_rt.group_id, next_decode_write_slot(layer_rt))
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
        aged_k, aged_v = read_slot_kv(
            key_cache,
            value_cache,
            slot,
            block_size=layer_rt.layout.block_size,
        )
        stash_aged_slot_kv(layer_rt, key=aged_k, value=aged_v)
    return slots or None


def iter_h2o_decode_slot_remaps(
    *,
    req_ids: Sequence[str],
    num_computed_tokens: Sequence[int] | np.ndarray,
    query_start_loc_np: np.ndarray,
    forward_ctx: Mapping[str, Any],
):
    """Yield ``(tok_idx, {group_id: write_slot})`` for decode tokens needing remap.

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
        slots = _stash_aged_slots_for_request(req_id=req_id, forward_ctx=forward_ctx)
        if slots is None:
            continue
        tok0 = int(query_start_loc_np[i])
        tok1 = int(query_start_loc_np[i + 1])
        if tok1 <= tok0:
            continue
        for tok_idx in range(tok0, tok1):
            yield tok_idx, slots


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

    Only remaps the KV cache groups that own the request's H2O layers, each
    with its own write slot, leaving Mamba/GDN groups untouched.
    """
    if not envs.VLLM_H2O:
        return
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

    logger.info(
        "[H2O_DIAG] remap n=%d first=(tok=%s slots=%s)",
        len(remaps),
        remaps[0][0],
        remaps[0][1],
    )
    num_groups = slot_mappings.shape[0]
    for tok_idx, slots in remaps:
        for g_idx, write_slot in slots.items():
            if g_idx < num_groups:
                slot_mappings[g_idx, tok_idx] = write_slot


def h2o_decode_after_commit(
    commits: dict[str, list[int]],
    forward_ctx: Mapping[str, Any],
) -> None:
    """Advance H2O state for committed decode tokens.

    Bi-window (``update_heavy=False``): circular recent rotate only — never
    Algorithm-1 FA decode scoring / promote-copy. Spec for this cut forbids
    every-decode heavy update; live KVarN can be mistaken for FA pages and
    AssertionError-kills the engine if scoring runs.

    Legacy Algorithm-1 (``update_heavy=True``) may still score on true FA pages.
    """
    if not envs.VLLM_H2O or not commits:
        return
    logger.info(
        "[H2O_DIAG] decode_after_commit begin n_reqs=%d positions=%s",
        len(commits),
        {k: v for k, v in commits.items()},
    )
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
            # Bi-window: policy-only rotate. Do not gate on layout heuristics.
            if not layer_rt.state.update_heavy:
                for pos in positions:
                    apply_committed_decode_step(
                        layer_rt,
                        new_pos=int(pos),
                        new_scores_delta={},
                    )
                layer_rt.pending_decode_q = None
                layer_rt.pending_decode_positions = []
                continue

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
                # Non-FA layouts skip FA rewrite even under Algorithm-1.
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
