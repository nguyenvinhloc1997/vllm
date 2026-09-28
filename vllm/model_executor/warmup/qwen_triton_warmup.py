# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Qwen Triton kernels from the loaded model's compile keys."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

_QWEN_MODEL_TYPES = frozenset(
    {
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)

# Covers L=1 constexpr, non-divisible runtime L, and divisible runtime L.
_FLA_POST_CONV_WARMUP_LENGTHS = (1, 2, 16)


@dataclass(frozen=True)
class _QwenGDNWarmupConfig:
    h: int
    hv: int
    k: int
    v: int
    conv_kernel_size: int
    conv_state: torch.Tensor
    conv_dtype: torch.dtype
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    state_stride_token: int
    state_dtype: torch.dtype

    @property
    def conv_dim(self) -> int:
        return 2 * self.h * self.k + self.hv * self.v


def _is_non_empty_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and value.numel() > 0


def _is_qwen_gdn_layer(module: object) -> bool:
    return all(
        hasattr(module, attr)
        for attr in (
            "num_k_heads",
            "num_v_heads",
            "head_k_dim",
            "head_v_dim",
            "conv_kernel_size",
            "tp_size",
            "kv_cache",
            "A_log",
            "dt_bias",
        )
    )


def _iter_qwen_gdn_layers(static_forward_context: object):
    if not isinstance(static_forward_context, dict):
        return

    for module in static_forward_context.values():
        if _is_qwen_gdn_layer(module):
            yield module


def _split_qwen_gdn_cache(kv_cache: object) -> tuple[torch.Tensor, torch.Tensor] | None:
    if isinstance(kv_cache, (list, tuple)) and len(kv_cache) >= 2:
        conv_cache, ssm_state = kv_cache[:2]
        if _is_non_empty_tensor(conv_cache) and _is_non_empty_tensor(ssm_state):
            return conv_cache, ssm_state

    if isinstance(kv_cache, torch.Tensor) and kv_cache.size(0) >= 2:
        conv_cache = kv_cache[0]
        ssm_state = kv_cache[1]
        if _is_non_empty_tensor(conv_cache) and _is_non_empty_tensor(ssm_state):
            return conv_cache, ssm_state
    return None


def _qwen_gdn_warmup_config(
    static_forward_context: object,
) -> _QwenGDNWarmupConfig | None:
    found_layer = False
    for layer in _iter_qwen_gdn_layers(static_forward_context):
        found_layer = True
        cache_tensors = _split_qwen_gdn_cache(getattr(layer, "kv_cache", None))
        if cache_tensors is None:
            continue

        conv_cache, ssm_state = cache_tensors
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        conv_state = (
            conv_cache if is_conv_state_dim_first() else conv_cache.transpose(-1, -2)
        )
        tp_size = int(layer.tp_size)
        h = int(layer.num_k_heads) // tp_size
        hv = int(layer.num_v_heads) // tp_size

        return _QwenGDNWarmupConfig(
            h=h,
            hv=hv,
            k=int(layer.head_k_dim),
            v=int(layer.head_v_dim),
            conv_kernel_size=int(layer.conv_kernel_size),
            conv_state=conv_state,
            conv_dtype=conv_state.dtype,
            a_log=layer.A_log,
            dt_bias=layer.dt_bias,
            state_stride_token=int(ssm_state.stride(0)),
            state_dtype=ssm_state.dtype,
        )

    if found_layer:
        logger.info("Skipping Qwen GDN Triton warmup: no bound Qwen GDN cache found.")
    else:
        logger.info("Skipping Qwen GDN Triton warmup: no Qwen GDN layer found.")
    return None


def _warm_causal_conv1d_fwd_kernel(
    device: torch.device, config: _QwenGDNWarmupConfig
) -> None:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

    x_storage = torch.empty(
        (1, config.conv_dim), dtype=config.conv_dtype, device=device
    )
    x = x_storage.t()
    weight = torch.empty(
        (config.conv_dim, config.conv_kernel_size),
        dtype=config.conv_dtype,
        device=device,
    )
    cache_indices = torch.full((1,), NULL_BLOCK_ID, dtype=torch.int32, device=device)
    has_initial_state = torch.empty(1, dtype=torch.bool, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)

    causal_conv1d_fn(
        x,
        weight,
        None,
        config.conv_state,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
        pad_slot_id=PAD_SLOT_ID,
        null_block_id=NULL_BLOCK_ID,
        metadata=None,
        validate_data=False,
    )


def _warm_fused_post_conv_kernel(
    device: torch.device, config: _QwenGDNWarmupConfig
) -> None:
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (  # noqa: E501
        fused_post_conv_prep,
    )

    qkv_dim = 2 * config.h * config.k + config.hv * config.v
    for length in _FLA_POST_CONV_WARMUP_LENGTHS:
        conv_output = torch.empty(
            (length, qkv_dim), dtype=config.conv_dtype, device=device
        )
        a = torch.empty((length, config.hv), dtype=config.conv_dtype, device=device)
        b = torch.empty_like(a)

        fused_post_conv_prep(
            conv_output,
            a,
            b,
            config.a_log,
            config.dt_bias,
            config.h,
            config.k,
            config.v,
            apply_l2norm=True,
            output_g_exp=False,
        )


def _warm_fused_sigmoid_gating_delta_rule_update_kernel(
    device: torch.device,
    config: _QwenGDNWarmupConfig,
) -> None:
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    q = torch.empty((1, 1, config.h, config.k), dtype=config.conv_dtype, device=device)
    k = torch.empty_like(q)
    v = torch.empty((1, 1, config.hv, config.v), dtype=config.conv_dtype, device=device)
    a = torch.empty((1, 1, config.hv), dtype=config.conv_dtype, device=device)
    b = torch.empty_like(a)
    state = torch.empty(
        (1, config.state_stride_token),
        dtype=config.state_dtype,
        device=device,
    )
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    ssm_state_indices = torch.empty((1, 1), dtype=torch.int32, device=device)
    ssm_state_indices.zero_()

    fused_sigmoid_gating_delta_rule_update(
        A_log=config.a_log,
        a=a,
        b=b,
        dt_bias=config.dt_bias,
        q=q,
        k=k,
        v=v,
        beta=1.0,
        threshold=20.0,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
    )


def _warm_gdn_recoverssm_kernels(
    runner: Any, static_forward_context: Any, device: torch.device
) -> None:
    """Compile the RecoverSSM verify-record, commit and align-postprocess
    variants a mixed prefill + spec-decode step would otherwise JIT on the
    request path. Real caches are touched only at the null block (0); the
    commit plan writes only a throwaway context's scratch."""
    from vllm.model_executor.layers.mamba.damp_gdn_update import damp_fused_update
    from vllm.model_executor.layers.mamba.gdn_recoverssm import (
        GDNRecoverSSMCommitContext,
    )
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
    from vllm.v1.worker.gpu.model_states.recoverssm import (
        _postprocess_recoverssm_align_kernel,
    )

    groups = runner.kv_cache_config.kv_cache_groups
    for gid, group in enumerate(groups):
        layers: list[Any] = [static_forward_context.get(n) for n in group.layer_names]
        if layers and all(_is_qwen_gdn_layer(layer) for layer in layers):
            break
    else:
        return
    layer = layers[0]
    kv = layer.kv_cache
    h = int(layer.num_k_heads) // int(layer.tp_size)
    hv = int(layer.num_v_heads) // int(layer.tp_size)
    s = 1 + int(layer.num_spec)
    dtype = kv[0].dtype  # conv state dtype == activation dtype

    def ints(*shape: int, fill: int = NULL_BLOCK_ID, dt=torch.int32):
        return torch.full(shape, fill, dtype=dt, device=device)

    # Verify in record mode: one spec row whose checkpoint is the null block.
    q = torch.zeros((1, s, h, int(layer.head_k_dim)), dtype=dtype, device=device)
    v = torch.zeros((1, s, hv, int(layer.head_v_dim)), dtype=dtype, device=device)
    a = torch.zeros((1, s, hv), dtype=dtype, device=device)
    spec_update = (
        damp_fused_update
        if kv[1].dtype == torch.uint8
        else fused_sigmoid_gating_delta_rule_update
    )
    spec_update(
        A_log=layer.A_log,
        a=a,
        b=a,
        dt_bias=layer.dt_bias,
        q=q,
        k=q,
        v=v,
        initial_state=kv[1],
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, s], dtype=torch.int32, device=device),
        ssm_state_indices=ints(1, 1),
        num_accepted_tokens=ints(1, fill=1),
        use_qk_l2norm_in_kernel=True,
        recoverssm_records=(kv[2], kv[3], kv[4]),
    )

    # Commit + align postprocess, without and with request_indices (the
    # mixed-batch variant). The null source makes every kernel return early.
    align = layer.cache_config.mamba_cache_mode == "align"
    block_size = int(group.kv_cache_spec.block_size)
    width = runner.block_tables.input_block_tables[gid].shape[1]
    ctx = GDNRecoverSSMCommitContext.create(layers, spec_query_len=s, max_num_reqs=1)
    num_sampled = ints(1, fill=s)
    num_computed = ints(1) if align else None
    for request_indices in (None, ints(1, fill=0)):
        ctx.commit(
            num_sampled,
            ints(1, 1)[:, 0],
            torch.tensor([0, s], dtype=torch.int32, device=device),
            request_indices=request_indices,
            block_table=ints(1, width) if align else None,
            num_computed_tokens=num_computed,
            mamba_block_size=block_size if align else None,
        )
        if align:
            _postprocess_recoverssm_align_kernel[(1,)](
                ints(1, fill=-1, dt=torch.int64),  # idx_mapping: no request
                num_sampled,
                request_indices,
                num_computed,
                ints(1),
                ints(1),
                MAMBA_BLOCK_SIZE=block_size,
                BLOCK_TABLE_WIDTH=width,
            )


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.accelerator.synchronize(device)


@torch.inference_mode()
def qwen_triton_warmup(
    runner: "GPUModelRunner",
    model_config: object,
) -> None:
    """Warm Qwen Triton kernels reported by the JIT monitor."""
    if runner.is_pooling_model:
        return

    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    model_type = None
    for config in (hf_text_config, hf_config):
        model_type = getattr(config, "model_type", None)
        if model_type is not None:
            model_type = str(model_type)
            break
    if model_type not in _QWEN_MODEL_TYPES:
        return

    device = getattr(runner, "device", torch.device("cuda"))
    logger.info("Warming up Qwen Triton kernels for model_type=%s.", model_type)

    compilation_config = getattr(runner, "compilation_config", None)
    static_forward_context = getattr(compilation_config, "static_forward_context", None)
    gdn_config = _qwen_gdn_warmup_config(static_forward_context)
    if gdn_config is None:
        return

    _warm_causal_conv1d_fwd_kernel(device, gdn_config)
    _warm_fused_post_conv_kernel(device, gdn_config)
    _warm_fused_sigmoid_gating_delta_rule_update_kernel(device, gdn_config)
    layer = next(_iter_qwen_gdn_layers(static_forward_context))
    if getattr(getattr(layer, "cache_config", None), "use_gdn_recoverssm", False):
        logger.info("Warming up GDN RecoverSSM Triton kernels.")
        _warm_gdn_recoverssm_kernels(runner, static_forward_context, device)
    _synchronize_device(device)
