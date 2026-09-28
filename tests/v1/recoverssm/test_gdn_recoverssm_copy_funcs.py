# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


@pytest.mark.parametrize("cls", [Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration])
@pytest.mark.parametrize("flag, n", [(False, 2), (True, 5)])
def test_copy_funcs_match_page_state_count(cls, flag, n):
    # The V2 align copier asserts one copy func per state tensor.
    model = cls.__new__(cls)
    model._use_gdn_recoverssm = flag
    funcs = model.get_mamba_state_copy_funcs({MambaAttentionBackendEnum.GDN_ATTN})
    assert len(funcs[MambaAttentionBackendEnum.GDN_ATTN]) == n
