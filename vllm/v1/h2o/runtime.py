# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/v1/h2o/runtime.py
"""Per-request H2O runtime (worker-side) for decode maintenance."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.v1.h2o.policy import H2OState
from vllm.v1.h2o.slots import SlotLayout


@dataclass
class H2OLayerRuntime:
    state: H2OState
    layout: SlotLayout


@dataclass
class H2ORequestRuntime:
    request_id: str
    prompt_len: int
    # layer_name → runtime; FA layers only.
    layers: dict[str, H2OLayerRuntime] = field(default_factory=dict)
    # Absolute positions of decode tokens committed since last maintenance.
    pending_committed_positions: list[int] = field(default_factory=list)


_RUNTIME: dict[str, H2ORequestRuntime] = {}


def get_h2o_runtime(request_id: str) -> H2ORequestRuntime | None:
    return _RUNTIME.get(request_id)


def set_h2o_layer_runtime(
    request_id: str,
    *,
    layer_name: str,
    state: H2OState,
    layout: SlotLayout,
    prompt_len: int,
) -> H2ORequestRuntime:
    rt = _RUNTIME.get(request_id)
    if rt is None:
        rt = H2ORequestRuntime(request_id=request_id, prompt_len=prompt_len)
        _RUNTIME[request_id] = rt
    rt.layers[layer_name] = H2OLayerRuntime(state=state, layout=layout)
    return rt


def clear_h2o_runtime(request_id: str) -> None:
    _RUNTIME.pop(request_id, None)


def clear_all_h2o_runtimes() -> None:
    _RUNTIME.clear()


def note_committed_decode_positions(request_id: str, positions: list[int]) -> None:
    """Record committed absolute positions for post-step maintenance."""
    rt = _RUNTIME.get(request_id)
    if rt is None:
        return
    rt.pending_committed_positions.extend(int(p) for p in positions)
