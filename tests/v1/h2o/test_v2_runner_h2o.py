# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# tests/v1/h2o/test_v2_runner_h2o.py
"""Import-level checks that V2 GPUModelRunner exposes H2O hooks."""

from __future__ import annotations


def test_v2_gpu_model_runner_has_h2o_decode_after_commit():
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    assert hasattr(GPUModelRunner, "h2o_decode_after_commit")
    assert callable(GPUModelRunner.h2o_decode_after_commit)


def test_v1_gpu_model_runner_still_has_h2o_decode_after_commit():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1GPUModelRunner

    assert hasattr(V1GPUModelRunner, "h2o_decode_after_commit")
    assert callable(V1GPUModelRunner.h2o_decode_after_commit)


def test_shared_runner_hooks_importable():
    from vllm.v1.h2o import runner_hooks

    assert callable(runner_hooks.h2o_decode_after_commit)
    assert callable(runner_hooks.maybe_set_h2o_batch_context)
    assert callable(runner_hooks.maybe_clamp_h2o_seq_lens)
    assert callable(runner_hooks.maybe_remap_h2o_decode_slots_gpu)
    assert callable(runner_hooks.maybe_clear_h2o_batch_context)
    assert callable(runner_hooks.maybe_clear_h2o_runtime)
