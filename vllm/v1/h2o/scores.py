# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/scores.py
from __future__ import annotations

import torch


def accumulate_attention_mass(
    q: torch.Tensor, k: torch.Tensor, scale: float
) -> torch.Tensor:
    # q: [T_q, H_q, D], k: [T_k, H_kv, D]
    t_q, h_q, d = q.shape
    t_k, h_kv, d_k = k.shape
    assert d == d_k and h_q % h_kv == 0
    group = h_q // h_kv
    # expand k to query heads
    k_exp = k.unsqueeze(1).expand(t_k, group, h_kv, d).reshape(t_k, h_q, d)
    # scores [T_q, H_q, T_k]
    logits = torch.einsum("qhd,khd->qhk", q * scale, k_exp)
    if t_q == t_k:
        mask = torch.triu(torch.ones(t_q, t_k, dtype=torch.bool, device=q.device), 1)
        logits = logits.masked_fill(mask.unsqueeze(1), float("-inf"))
    weights = torch.softmax(logits.float(), dim=-1).to(q.dtype)
    # sum queries → [H_q, T_k]; mean GQA group → [H_kv, T_k]; sum heads → [T_k]
    per_q_head = weights.sum(dim=0)
    per_kv = per_q_head.view(h_kv, group, t_k).mean(dim=1)
    return per_kv.sum(dim=0)


def mass_dict(mass: torch.Tensor, positions: list[int]) -> dict[int, float]:
    assert mass.numel() == len(positions)
    return {int(p): float(m) for p, m in zip(positions, mass.tolist())}
