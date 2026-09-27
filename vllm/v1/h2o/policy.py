# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/policy.py
from __future__ import annotations

from dataclasses import dataclass, field


def compute_k(prompt_len: int, ratio: float = 0.2) -> int:
    if prompt_len < 2:
        return 0
    total = int(ratio * prompt_len)
    if total < 2:
        total = 2
    if total % 2:
        total -= 1
    return total // 2


@dataclass
class H2OState:
    k: int
    heavy: list[int]
    recent: list[int]
    scores: dict[int, float] = field(default_factory=dict)
    update_heavy: bool = True


def select_prefill(scores: dict[int, float], prompt_len: int, k: int) -> H2OState:
    if k <= 0:
        return H2OState(k=0, heavy=[], recent=[], scores={})
    if prompt_len < 2 * k:
        positions = list(range(prompt_len))
        return H2OState(
            k=k,
            heavy=[],
            recent=positions,
            scores={p: scores.get(p, 0.0) for p in positions},
        )
    recent = list(range(prompt_len - k, prompt_len))
    prefix = list(range(0, prompt_len - k))
    prefix.sort(key=lambda p: (-scores.get(p, 0.0), p))
    heavy = sorted(prefix[:k])
    keep = set(heavy) | set(recent)
    return H2OState(
        k=k,
        heavy=heavy,
        recent=recent,
        scores={p: scores.get(p, 0.0) for p in keep},
    )


def decode_step(
    state: H2OState, new_pos: int, new_scores_delta: dict[int, float]
) -> H2OState:
    if state.k <= 0:
        return state
    if not state.update_heavy:
        recent = [*state.recent[1:], new_pos]
        keep = set(state.heavy) | set(recent)
        scores = {p: state.scores.get(p, 0.0) for p in keep}
        return H2OState(
            k=state.k,
            heavy=list(state.heavy),
            recent=recent,
            scores=scores,
            update_heavy=False,
        )
    scores = dict(state.scores)
    for p, d in new_scores_delta.items():
        if p in scores or p == new_pos:
            scores[p] = scores.get(p, 0.0) + d
    recent = list(state.recent)
    heavy = list(state.heavy)
    if len(recent) < state.k:
        recent.append(new_pos)
        scores.setdefault(new_pos, 0.0)
        return H2OState(k=state.k, heavy=heavy, recent=recent, scores=scores)
    old = recent.pop(0)
    recent.append(new_pos)
    scores.setdefault(new_pos, 0.0)
    candidates = heavy + [old]
    # lowest score wins eviction; tie → higher position drops
    drop = min(candidates, key=lambda p: (scores.get(p, 0.0), -p))
    heavy = [p for p in candidates if p != drop]
    keep = set(heavy) | set(recent)
    scores = {p: scores.get(p, 0.0) for p in keep}
    return H2OState(k=state.k, heavy=sorted(heavy), recent=recent, scores=scores)
