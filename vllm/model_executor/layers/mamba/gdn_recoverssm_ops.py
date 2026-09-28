# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton helpers shared by the GDN verify kernels and the RecoverSSM commit.

One source for the rank-1 update and the DAMP int8 quantizer keeps the
compiled arithmetic identical on both sides, so the committed state can match
the baseline's per-draft state bit for bit.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def _gdn_rank1(b_h, b_c, b_k):
    return b_h + b_c[:, None] * b_k[None, :]


@triton.jit
def _damp_quantize(b_lo):
    amax = tl.max(tl.abs(b_lo))
    sc = tl.maximum(amax, 1e-8) / 127.0
    qv = b_lo / sc
    codes = tl.where(qv >= 0, tl.floor(qv + 0.5), tl.ceil(qv - 0.5))
    codes = tl.minimum(tl.maximum(codes, -127.0), 127.0)
    return codes, sc
