# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)


def test_gdn_recoverssm_record_shapes_and_dtypes():
    base_shapes = ((10240, 10), (48, 128, 128))
    shapes = MambaStateShapeCalculator.append_gdn_recoverssm_record(
        base_shapes,
        num_v_heads=48,
        head_k_dim=128,
        head_v_dim=128,
        tp_world_size=1,
        spec_query_len=8,
    )
    assert shapes == (*base_shapes, (48, 8, 128), (48, 8, 128), (48, 8))

    dtypes = MambaStateDtypeCalculator.append_gdn_recoverssm_record(
        (torch.bfloat16, torch.float16)
    )
    assert dtypes == (
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float32,
        torch.float32,
    )


def test_record_bytes_match_spec():
    # 196,608 + 196,608 + 1,536 = 394,752 B per layer at Hv=48, K=V=128, S=8
    shapes = MambaStateShapeCalculator.append_gdn_recoverssm_record(
        ((1, 1), (1, 1, 1)),
        num_v_heads=48,
        head_k_dim=128,
        head_v_dim=128,
        tp_world_size=1,
        spec_query_len=8,
    )[2:]
    nbytes = sum(4 * torch.Size(s).numel() for s in shapes)
    assert nbytes == 394_752
