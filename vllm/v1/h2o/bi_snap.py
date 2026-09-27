# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/bi_snap.py
"""Bi-window SnapKV-style prefill select + recent (no hard pin)."""

from __future__ import annotations

from vllm.v1.h2o.policy import H2OState


def bi_snap_budget(
    prompt_len: int, *, ratio: float = 0.4, w: int = 256
) -> tuple[int, int, int]:
    """Return ``(keep, R, M)`` with ``R = max(w, keep//2)`` capped by prompt."""
    if prompt_len <= 0:
        return 0, 0, 0
    keep = max(1, int(ratio * prompt_len))
    keep = min(keep, prompt_len)
    r = min(prompt_len, max(w, keep // 2))
    m = max(0, keep - r)
    return keep, r, m


def select_bi_snap(
    mass: dict[int, float],
    prompt_len: int,
    *,
    ratio: float = 0.4,
    w0: int = 256,
    w: int = 256,
    keep: int | None = None,
    recent_len: int | None = None,
    middle_budget: int | None = None,
) -> H2OState:
    """Keep last ``R`` plus top-``M`` of ``[0, T-R)`` by ``mass``.

    ``w0`` is unused at select time (observation windows only affect how ``mass``
    was built). ``heavy`` holds Snap middle winners; ``recent`` holds the tail.
    """
    del w0  # mass already incorporates both windows
    if prompt_len <= 0:
        return H2OState(k=0, heavy=[], recent=[], scores={})

    keep_n, r, m = bi_snap_budget(prompt_len, ratio=ratio, w=w)
    if keep is not None:
        keep_n = min(max(1, keep), prompt_len)
        r = min(prompt_len, max(w, keep_n // 2))
        m = max(0, keep_n - r)
    if recent_len is not None:
        r = min(prompt_len, max(0, recent_len))
    if middle_budget is not None:
        m = max(0, middle_budget)

    if r >= prompt_len or keep_n >= prompt_len:
        positions = list(range(prompt_len))
        return H2OState(
            k=0,
            heavy=[],
            recent=positions,
            scores={p: float(mass.get(p, 0.0)) for p in positions},
        )

    recent = list(range(prompt_len - r, prompt_len))
    candidates = list(range(0, prompt_len - r))
    candidates.sort(key=lambda p: (-float(mass.get(p, 0.0)), p))
    heavy = sorted(candidates[:m]) if m > 0 else []
    keep_set = set(heavy) | set(recent)
    return H2OState(
        k=m,
        heavy=heavy,
        recent=recent,
        scores={p: float(mass.get(p, 0.0)) for p in keep_set},
    )
