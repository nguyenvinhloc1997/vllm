# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec gate G1: RecoverSSM conv compaction matches the baseline conv window.

Runs the real GDN layer's spec conv call (``_forward_core``) in record mode,
compacts with the commit's conv kernel, and checks that the conv history the
next step reads is bitwise equal to what the baseline (per-draft) path's next
step reads at offset ``num_accepted - 1``.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_mod
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.models.kimi_k3.nvidia.ops.recoverssm import _compact_conv_state_kernel
from vllm.triton_utils import triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DIM, WIDTH, NUM_SPEC = 128, 4, 7
S = 1 + NUM_SPEC
STATE_LEN = WIDTH - 1 + NUM_SPEC
SLOT = 1


class _Stop(Exception):
    pass


def _conv_cache(dev):
    # Returns (kv_cache[0] storage, (blocks, dim, state_len) view).
    if is_conv_state_dim_first():
        t = torch.randn(3, DIM, STATE_LEN, device=dev, dtype=torch.bfloat16)
        return t, t
    t = torch.randn(3, STATE_LEN, DIM, device=dev, dtype=torch.bfloat16)
    return t, t.transpose(-1, -2)


def _layer_record_conv(x, w, cache):
    """Run the layer's spec conv call with RecoverSSM on; stop after it."""
    dev = x.device
    ones = torch.ones(1, dtype=torch.int32, device=dev)
    md = object.__new__(GDNAttentionMetadata)
    md.__dict__.update(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=1,
        num_actual_tokens=S,
        num_accepted_tokens=ones,
        has_initial_state=None,
        spec_query_start_loc=torch.tensor([0, S], dtype=torch.int32, device=dev),
        non_spec_query_start_loc=None,
        spec_sequence_masks=torch.ones(1, dtype=torch.bool, device=dev),
        spec_token_indx=None,
        non_spec_token_indx=None,
        spec_state_indices_tensor=torch.tensor([[SLOT]], dtype=torch.int32, device=dev),
        non_spec_state_indices_tensor=None,
    )
    layer = SimpleNamespace(
        prefix="gdn",
        enable_packed_recurrent_decode=False,
        kv_cache=(cache, None),
        conv1d=SimpleNamespace(weight=w.view(DIM, 1, WIDTH), bias=None),
        activation="silu",
        num_spec=NUM_SPEC,
        cache_config=SimpleNamespace(use_gdn_recoverssm=True),
    )

    def spy(*args, **kwargs):
        causal_conv1d_update(*args, **kwargs)
        raise _Stop

    ctx = SimpleNamespace(attn_metadata={"gdn": md})
    mp = pytest.MonkeyPatch()
    mp.setattr(gdn_mod, "get_forward_context", lambda: ctx)
    mp.setattr(gdn_mod, "causal_conv1d_update", spy)
    try:
        with pytest.raises(_Stop):
            gdn_mod.QwenGatedDeltaNetAttention._forward_core(
                layer, x, x.new_zeros(S, 1), x.new_zeros(S, 1), None
            )
    finally:
        mp.undo()


def _compact(view, commit_len):
    dev = view.device
    i32 = lambda v: torch.tensor([v], dtype=torch.int32, device=dev)  # noqa: E731
    i64 = lambda v: torch.tensor([v], dtype=torch.int64, device=dev)  # noqa: E731
    history = STATE_LEN - S + 1
    _compact_conv_state_kernel[(triton.cdiv(DIM, 256), 1, 1)](
        view,
        i64(view.data_ptr()),
        i64(view.stride(0)),
        i64(view.stride(1)),
        i64(view.stride(2)),
        i32(SLOT),
        i32(commit_len),
        i32(SLOT),
        i32(NULL_BLOCK_ID),
        i32(0),
        NULL_BLOCK_ID,
        DIM,
        history,
        1,
        BLOCK_D=256,
        BLOCK_HISTORY=triton.next_power_of_2(history),
        ALIGN_MODE=False,
        num_warps=4,
    )


@pytest.mark.parametrize("num_accepted", range(1, S + 1))
def test_record_conv_compaction_matches_baseline_window(num_accepted):
    torch.manual_seed(num_accepted)
    dev = "cuda"
    x = torch.randn(S, DIM, device=dev, dtype=torch.bfloat16)
    w = torch.randn(DIM, WIDTH, device=dev, dtype=torch.bfloat16)
    cache, view = _conv_cache(dev)
    base_cache = cache.clone()
    base_view = (
        base_cache if base_cache.shape[1] == DIM else base_cache.transpose(-1, -2)
    )

    # Baseline (per-draft slots, [N, S] indices): max_query_len=S; the next
    # step reads its conv window at offset num_accepted - 1.
    causal_conv1d_update(
        x.clone(),
        base_view,
        w,
        None,
        "silu",
        conv_state_indices=torch.tensor([SLOT], dtype=torch.int32, device=dev),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=dev),
        query_start_loc=torch.tensor([0, S], dtype=torch.int32, device=dev),
        max_query_len=S,
        validate_data=False,
    )
    expected = base_view[SLOT, :, num_accepted - 1 : num_accepted - 1 + WIDTH - 1]

    # Record: the layer's own call, then commit compaction; next step reads
    # offset 0 (num_accepted_tokens is 1 in record mode).
    _layer_record_conv(x.clone(), w, cache)
    _compact(view, num_accepted)
    got = view[SLOT, :, : WIDTH - 1]

    assert torch.equal(got, expected)
