# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/compress.py
"""Prefill KV compress orchestration for H2O (Algorithm 1).

Production scoring uses S2 chunked mass (never ``T×T``). Ownership-A block
table resize lives on the KV manager; this module selects, gathers, and can
repack kept K/V into the leading pages of a paged cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm import envs
from vllm.utils.math_utils import cdiv
from vllm.v1.h2o.ownership import num_keep_tokens
from vllm.v1.h2o.policy import H2OState, compute_k, select_prefill
from vllm.v1.h2o.scores import accumulate_attention_mass_chunked


def should_compress_prefill(
    *,
    is_full_attention: bool,
    is_last_prefill_chunk: bool,
    num_computed_tokens: int | None = None,
    prompt_len: int | None = None,
) -> bool:
    """True when end-of-prefill H2O compress should run for this layer/step.

    Prefer ``num_computed_tokens == prompt_len`` when both are supplied; the
    boolean ``is_last_prefill_chunk`` covers callers that already derived it.
    """
    if not envs.VLLM_H2O or not is_full_attention:
        return False
    if num_computed_tokens is not None and prompt_len is not None:
        return num_computed_tokens == prompt_len and prompt_len > 0
    return bool(is_last_prefill_chunk)


def compress_prefill_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: list[int],
    ratio: float = 0.2,
    *,
    mass: torch.Tensor | None = None,
    tile_q: int = 64,
    tile_k: int = 64,
) -> tuple[H2OState, torch.Tensor, torch.Tensor, list[int]]:
    """Score (S2), select, and gather prefill K/V keeping absolute positions.

    Args:
        q: Query tensor ``[T, H_q, D]``.
        k: Key tensor ``[T, H_kv, D]``.
        v: Value tensor ``[T, H_kv, D]``.
        positions: Absolute RoPE positions length ``T`` (not necessarily 0..T-1).
        ratio: Total keep fraction (even split into heavy + recent).
        mass: Optional pre-accumulated per-key mass ``[T]`` (skips scoring).
        tile_q: S2 query tile size.
        tile_k: S2 key tile size.

    Returns:
        ``(state, k_out, v_out, positions_out)`` where ``positions_out`` is the
        kept absolute positions in causal order, and ``state.heavy`` /
        ``state.recent`` use those same absolute positions.
    """
    t = k.shape[0]
    if len(positions) != t:
        raise ValueError(f"positions length {len(positions)} != key length {t}")
    if v.shape[0] != t:
        raise ValueError("k, v must share the same sequence length")
    if mass is None:
        if q.shape[0] != t:
            raise ValueError("q, k, v must share the same sequence length")
        scale = q.shape[-1] ** -0.5
        mass = accumulate_attention_mass_chunked(
            q,
            k,
            scale=scale,
            tile_q=tile_q,
            tile_k=tile_k,
            q_positions=positions,
            k_positions=positions,
        )
    elif mass.numel() != t:
        raise ValueError(f"mass length {mass.numel()} != key length {t}")

    local_scores = {i: float(mass[i].item()) for i in range(t)}
    k_budget = compute_k(t, ratio)
    local_state = select_prefill(local_scores, t, k_budget)

    heavy = [positions[i] for i in local_state.heavy]
    recent = [positions[i] for i in local_state.recent]
    scores = {positions[i]: s for i, s in local_state.scores.items()}
    state = H2OState(k=local_state.k, heavy=heavy, recent=recent, scores=scores)

    keep_local = sorted(set(local_state.heavy) | set(local_state.recent))
    positions_out = [positions[i] for i in keep_local]
    return state, k[keep_local], v[keep_local], positions_out


def try_compress_prefill_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: list[int],
    *,
    is_full_attention: bool,
    is_last_prefill_chunk: bool,
    num_computed_tokens: int | None = None,
    prompt_len: int | None = None,
    ratio: float | None = None,
    mass: torch.Tensor | None = None,
) -> tuple[H2OState, torch.Tensor, torch.Tensor, list[int]] | None:
    """Flag-gated wrapper around :func:`compress_prefill_kv`."""
    if not should_compress_prefill(
        is_full_attention=is_full_attention,
        is_last_prefill_chunk=is_last_prefill_chunk,
        num_computed_tokens=num_computed_tokens,
        prompt_len=prompt_len,
    ):
        return None
    if ratio is None:
        ratio = float(envs.VLLM_H2O_RATIO)
    return compress_prefill_kv(q, k, v, positions, ratio=ratio, mass=mass)


def is_full_attention_sliding_window(
    sliding_window: tuple[int, int] | None,
) -> bool:
    """True for global / full-attention layers (no finite sliding window)."""
    if sliding_window is None:
        return True
    return sliding_window[0] < 0 and sliding_window[1] < 0


@dataclass
class PrefillMassAccumulator:
    """Accumulate S2 mass across chunked-prefill tiles for one request/layer.

    Buffers K/V by absolute (or local) position so the last chunk can compress
    from the full prompt without a second page gather when callers stream
    chunk tensors. Mass for each chunk is scored against all keys buffered so
    far (causal via absolute positions).
    """

    prompt_len: int
    mass: torch.Tensor = field(init=False)
    k_buf: torch.Tensor | None = field(default=None, init=False, repr=False)
    v_buf: torch.Tensor | None = field(default=None, init=False, repr=False)
    filled: torch.Tensor = field(init=False, repr=False)
    # Physical FA block ids observed across chunks (order preserved).
    block_ids: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.mass = torch.zeros(self.prompt_len, dtype=torch.float32)
        self.filled = torch.zeros(self.prompt_len, dtype=torch.bool)
        self.block_ids = []

    def note_slots(self, slots: list[int], *, block_size: int) -> None:
        """Record physical block ids touched by this chunk's slot_mapping."""
        seen = set(self.block_ids)
        for s in slots:
            if s < 0:
                continue
            bid = int(s) // block_size
            if bid not in seen:
                seen.add(bid)
                self.block_ids.append(bid)

    def _ensure_kv_buffers(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if self.k_buf is not None:
            return
        h, d = k.shape[1], k.shape[2]
        # Host-side buffers: live CTX=huge already pins the 3090; keeping a
        # full-prompt K/V copy per FA layer on GPU OOMs during chunked prefill.
        self.k_buf = torch.zeros(self.prompt_len, h, d, device="cpu", dtype=k.dtype)
        self.v_buf = torch.zeros(self.prompt_len, h, d, device="cpu", dtype=v.dtype)

    def _local_index(self, pos: int, order_i: int) -> int | None:
        if 0 <= pos < self.prompt_len:
            return pos
        if 0 <= order_i < self.prompt_len:
            return order_i
        return None

    def buffer_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: list[int],
    ) -> None:
        """Store this chunk's K/V into the prompt-sized buffers (CPU)."""
        if len(positions) != k.shape[0] or k.shape[0] != v.shape[0]:
            raise ValueError("k/v/positions length mismatch")
        self._ensure_kv_buffers(k, v)
        assert self.k_buf is not None and self.v_buf is not None
        k_cpu = k.detach().to("cpu")
        v_cpu = v.detach().to("cpu")
        for i, pos in enumerate(positions):
            idx = self._local_index(int(pos), i)
            if idx is None:
                continue
            self.k_buf[idx].copy_(k_cpu[i])
            self.v_buf[idx].copy_(v_cpu[i])
            self.filled[idx] = True

    def update(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        q_positions: list[int],
        k_positions: list[int],
        scale: float | None = None,
        tile_q: int = 64,
        tile_k: int = 64,
        v: torch.Tensor | None = None,
    ) -> None:
        if scale is None:
            scale = q.shape[-1] ** -0.5
        if len(k_positions) > self.prompt_len:
            raise ValueError("k_positions exceed prompt_len")
        if v is not None:
            self.buffer_kv(k, v, k_positions)
            # Score chunk Q against all keys buffered so far (both on CPU).
            filled_idx = self.filled.nonzero(as_tuple=False).flatten().tolist()
            if not filled_idx:
                return
            assert self.k_buf is not None
            q_cpu = q.detach().to("cpu")
            k_all = self.k_buf[filled_idx]
            chunk_mass = accumulate_attention_mass_chunked(
                q_cpu,
                k_all,
                scale=scale,
                tile_q=tile_q,
                tile_k=tile_k,
                q_positions=q_positions,
                k_positions=filled_idx,
            )
            for i, pos in enumerate(filled_idx):
                self.mass[pos] += chunk_mass[i].to(self.mass.device)
            return

        q_cpu = q.detach().to(self.mass.device)
        k_cpu = k.detach().to(self.mass.device)
        chunk_mass = accumulate_attention_mass_chunked(
            q_cpu,
            k_cpu,
            scale=scale,
            tile_q=tile_q,
            tile_k=tile_k,
            q_positions=q_positions,
            k_positions=k_positions,
        )
        for i, pos in enumerate(k_positions):
            idx = self._local_index(int(pos), i)
            if idx is not None:
                self.mass[idx] += chunk_mass[i]

    def full_kv(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]] | None:
        """Return dense prompt K/V when every position has been buffered."""
        if self.k_buf is None or self.v_buf is None:
            return None
        if not bool(self.filled.all().item()):
            return None
        positions = list(range(self.prompt_len))
        return self.k_buf, self.v_buf, positions


def repack_kv_into_pages(
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    *,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_ids: list[int],
    block_size: int,
) -> int:
    """Write gathered K/V densely into the leading pages of a paged FA cache.

    Args:
        k_out: ``[T_keep, H_kv, D]``.
        v_out: ``[T_keep, H_kv, D]``.
        key_cache: ``[num_blocks, block_size, H_kv, D]`` (FA layout after split).
        value_cache: Same shape as ``key_cache``.
        block_ids: Physical block ids for the request (post-resize prefix).
        block_size: Tokens per block.

    Returns:
        Number of blocks written into.
    """
    t_keep = k_out.shape[0]
    n_blocks = cdiv(t_keep, block_size)
    if len(block_ids) < n_blocks:
        raise ValueError(
            f"need {n_blocks} blocks to pack {t_keep} tokens, got {len(block_ids)}"
        )
    for bi in range(n_blocks):
        start = bi * block_size
        end = min(start + block_size, t_keep)
        length = end - start
        bid = block_ids[bi]
        key_cache[bid, :length].copy_(k_out[start:end])
        value_cache[bid, :length].copy_(v_out[start:end])
        if length < block_size:
            key_cache[bid, length:].zero_()
            value_cache[bid, length:].zero_()
    return n_blocks


def after_full_attention_kv_update(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor | None = None,
    positions: list[int] | None = None,
    is_last_prefill_chunk: bool = False,
    num_computed_tokens: int | None = None,
    prompt_len: int | None = None,
    sliding_window: tuple[int, int] | None = None,
    ratio: float | None = None,
    mass: torch.Tensor | None = None,
) -> tuple[H2OState, torch.Tensor, torch.Tensor, list[int]] | None:
    """Hook (b) entry: after full-attention KV pages are written.

    Runs :func:`try_compress_prefill_kv` when the runner supplies end-of-prefill
    ``query`` (or ``mass``) + absolute ``positions`` and the prompt is fully
    computed. Ownership-A block-table resize is applied by the KV manager after
    the worker step (deterministic ``2K`` from prompt length).
    """
    if positions is None:
        return None
    if query is None and mass is None:
        return None
    # Dummy q when mass is pre-accumulated (page-gather path).
    if query is None:
        assert mass is not None
        query = torch.zeros(
            mass.shape[0],
            1,
            key.shape[-1],
            device=key.device,
            dtype=key.dtype,
        )
    return try_compress_prefill_kv(
        query,
        key,
        value,
        positions,
        is_full_attention=is_full_attention_sliding_window(sliding_window),
        is_last_prefill_chunk=is_last_prefill_chunk,
        num_computed_tokens=num_computed_tokens,
        prompt_len=prompt_len,
        ratio=ratio,
        mass=mass,
    )


def run_prefill_compress_for_request(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: list[int],
    prompt_len: int,
    num_computed_tokens: int,
    is_full_attention: bool,
    kv_cache_manager=None,
    request_id: str | None = None,
    ratio: float | None = None,
    key_cache: torch.Tensor | None = None,
    value_cache: torch.Tensor | None = None,
    block_ids: list[int] | None = None,
    block_size: int | None = None,
    mass: torch.Tensor | None = None,
) -> tuple[H2OState, torch.Tensor, torch.Tensor, list[int]] | None:
    """End-of-prefill orchestration: compress, optional page repack, optional resize.

    Used by the manager-level integration test and by the live runner path.
    """
    if ratio is None:
        ratio = float(envs.VLLM_H2O_RATIO)
    out = try_compress_prefill_kv(
        q,
        k,
        v,
        positions,
        is_full_attention=is_full_attention,
        is_last_prefill_chunk=True,
        num_computed_tokens=num_computed_tokens,
        prompt_len=prompt_len,
        ratio=ratio,
        mass=mass,
    )
    if out is None:
        return None
    state, k_out, v_out, pos_out = out

    if (
        key_cache is not None
        and value_cache is not None
        and block_ids is not None
        and block_size is not None
    ):
        keep_n = num_keep_tokens(prompt_len, ratio)
        keep_blocks = cdiv(keep_n, block_size)
        repack_kv_into_pages(
            k_out,
            v_out,
            key_cache=key_cache,
            value_cache=value_cache,
            block_ids=block_ids[:keep_blocks],
            block_size=block_size,
        )

    if kv_cache_manager is not None and request_id is not None:
        kv_cache_manager.resize_h2o_full_attention(
            request_id, num_keep_tokens(prompt_len, ratio)
        )
    return state, k_out, v_out, pos_out
