# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G1 exactness: RecoverSSM record mode + commit vs the per-draft baseline."""

import pytest
import torch

from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")

HV, H, K, V, S = 48, 16, 128, 128, 8
DEV = "cuda"


def _inputs(n_req: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    total = n_req * S

    def r(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=g, device=DEV) * scale).to(dtype)

    return dict(
        A_log=r(HV, dtype=torch.float32, scale=0.5),
        a=r(total, HV),
        b=r(total, HV),
        dt_bias=r(HV, dtype=torch.float32, scale=0.5),
        q=r(1, total, H, K),
        k=r(1, total, H, K),
        v=r(1, total, HV, V),
        cu_seqlens=torch.arange(0, total + 1, S, device=DEV, dtype=torch.int32),
    )


def _records(num_blocks: int):
    return (
        torch.zeros(num_blocks, HV, S, V, device=DEV, dtype=torch.float32),
        torch.zeros(num_blocks, HV, S, K, device=DEV, dtype=torch.float32),
        torch.zeros(num_blocks, HV, S, device=DEV, dtype=torch.float32),
    )


def _plain_baseline_and_record(n_req, seed, dtype):
    """Baseline: per-draft slots 1+r*S .. . Record: checkpoint at 1+r*S."""
    inp = _inputs(n_req, seed)
    num_blocks = 1 + n_req * S
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    state0 = (torch.randn(num_blocks, HV, V, K, generator=g, device=DEV) * 0.1).to(
        dtype
    )
    state0[0] = 0

    base_state = state0.clone()
    base_idx = torch.arange(1, num_blocks, device=DEV, dtype=torch.int32).view(n_req, S)
    ones = torch.ones(n_req, device=DEV, dtype=torch.int32)
    o_base, _ = fused_sigmoid_gating_delta_rule_update(
        initial_state=base_state,
        ssm_state_indices=base_idx,
        num_accepted_tokens=ones,
        use_qk_l2norm_in_kernel=True,
        **inp,
    )

    rec_state = state0.clone()
    rec = _records(num_blocks)
    rec_idx = base_idx[:, :1].contiguous()
    o_rec, _ = fused_sigmoid_gating_delta_rule_update(
        initial_state=rec_state,
        ssm_state_indices=rec_idx,
        num_accepted_tokens=ones,
        use_qk_l2norm_in_kernel=True,
        recoverssm_records=rec,
        **inp,
    )
    return state0, base_state, base_idx, o_base, rec_state, rec, rec_idx, o_rec


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_plain_record_mode_output_bitwise(dtype):
    state0, _, _, o_base, rec_state, rec, rec_idx, o_rec = _plain_baseline_and_record(
        n_req=2, seed=0, dtype=dtype
    )
    assert torch.equal(o_rec, o_base)
    assert torch.equal(rec_state, state0)  # record mode never stores state
    assert rec[0][rec_idx.flatten().long()].abs().sum() > 0
    assert rec[2][rec_idx.flatten().long()].abs().sum() > 0
