# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Boot warmup of the GDN RecoverSSM kernels touches only the null block."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.recoverssm.test_gdn_recoverssm_kernels import (  # noqa: F401
    DEV,
    HV,
    K,
    S,
    V,
    _conv,
    _damp_page,
    _records,
    damp_env,
)
from vllm.model_executor.warmup.qwen_triton_warmup import (
    _warm_gdn_recoverssm_kernels,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")

NUM_BLOCKS = 3


def _run_warmup(state: torch.Tensor, mode: str) -> tuple[torch.Tensor, ...]:
    g = torch.Generator(device=DEV).manual_seed(3)
    rec = tuple(t.normal_(generator=g) for t in _records(NUM_BLOCKS))
    kv = (_conv(NUM_BLOCKS, 4), state, *rec)
    layer = SimpleNamespace(
        num_k_heads=16,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        conv_kernel_size=4,
        tp_size=1,
        kv_cache=kv,
        A_log=torch.zeros(HV, device=DEV),
        dt_bias=torch.zeros(HV, device=DEV),
        num_spec=S - 1,
        cache_config=SimpleNamespace(mamba_cache_mode=mode, use_gdn_recoverssm=True),
    )
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["gdn"], kv_cache_spec=SimpleNamespace(block_size=16)
                )
            ]
        ),
        block_tables=SimpleNamespace(
            input_block_tables=[torch.zeros(4, 8, dtype=torch.int32, device=DEV)]
        ),
    )
    before = tuple(t[1:].clone() for t in kv)
    _warm_gdn_recoverssm_kernels(runner, {"gdn": layer}, torch.device(DEV))
    torch.accelerator.synchronize()
    for got, want in zip(kv, before):
        assert torch.equal(got[1:], want)
    return kv


@pytest.mark.parametrize("mode", ["none", "align"])
def test_warmup_plain_leaves_real_blocks(mode):
    g = torch.Generator(device=DEV).manual_seed(1)
    state = (torch.randn(NUM_BLOCKS, HV, V, K, generator=g, device=DEV) * 0.1).to(
        torch.bfloat16
    )
    _run_warmup(state, mode)


@pytest.mark.usefixtures("damp_env")
@pytest.mark.parametrize("mode", ["none", "align"])
def test_warmup_damp_leaves_real_blocks(mode):
    _run_warmup(_damp_page(NUM_BLOCKS, 2), mode)
