# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton helpers shared by the GDN verify kernels and the RecoverSSM commit.

One source for the rank-1 update and the DAMP int8 quantizer keeps the
compiled arithmetic identical on both sides, so the committed state can match
the baseline's per-draft state bit for bit.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gdn_rank1(b_h, b_c, b_k):
    # Explicit fma: with `b_h + b_c * b_k` LLVM contracts whichever product has
    # fewer uses, so `b_h * decay` (multi-use in verify, single-use in commit)
    # would round differently on the two sides.
    return tl.fma(b_c[:, None], b_k[None, :], b_h)


@triton.jit
def _damp_quantize(b_lo):
    amax = tl.max(tl.abs(b_lo))
    sc = tl.maximum(amax, 1e-8) / 127.0
    qv = b_lo / sc
    codes = tl.where(qv >= 0, tl.floor(qv + 0.5), tl.ceil(qv - 0.5))
    codes = tl.minimum(tl.maximum(codes, -127.0), 127.0)
    return codes, sc


def check_recoverssm_records(
    ssm_state_indices, records, num_heads: int, key_dim: int, value_dim: int
) -> None:
    """Host-side record-mode preconditions (shapes and dtypes only, no sync)."""
    if ssm_state_indices.ndim != 2 or ssm_state_indices.shape[1] != 1:
        raise ValueError(
            "RecoverSSM record mode needs one checkpoint index per request: "
            f"ssm_state_indices must be [N, 1], got {tuple(ssm_state_indices.shape)}"
        )
    rec_c, rec_k, rec_d = records
    for name, t in (("rec_c", rec_c), ("rec_k", rec_k), ("rec_d", rec_d)):
        if t.dtype != torch.float32:
            raise ValueError(f"RecoverSSM {name} must be float32, got {t.dtype}")
    if not rec_c.shape[2] == rec_k.shape[2] == rec_d.shape[2]:
        raise ValueError(
            "RecoverSSM records disagree on S: "
            f"rec_c {rec_c.shape[2]}, rec_k {rec_k.shape[2]}, rec_d {rec_d.shape[2]}"
        )
    if not rec_c.shape[1] == rec_k.shape[1] == rec_d.shape[1] == num_heads:
        raise ValueError(
            f"RecoverSSM records need {num_heads} value heads: rec_c "
            f"{rec_c.shape[1]}, rec_k {rec_k.shape[1]}, rec_d {rec_d.shape[1]}"
        )
    if rec_c.shape[3] != value_dim or rec_k.shape[3] != key_dim:
        raise ValueError(
            f"RecoverSSM records need V={value_dim}, K={key_dim}: "
            f"rec_c {rec_c.shape[3]}, rec_k {rec_k.shape[3]}"
        )
