# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.models.ngm_residual import (
    NgramResidual,
    fill_ngm_prev_ids,
)

D = 16


def _memory() -> NgramResidual:
    torch.manual_seed(0)
    return NgramResidual(torch.nn.Embedding(50, D), embedding_dim=D)


def _reference_ngrams(embed, ids: torch.Tensor) -> torch.Tensor:
    # Original PioneerQyw math on one sequence: zero-padded avg_pool per n.
    t = embed(ids).T.unsqueeze(0)
    outs = [
        F.avg_pool1d(F.pad(t, (n - 1, 0)), kernel_size=n, stride=1)[0].T for n in (2, 3)
    ]
    return torch.stack(outs, dim=1)


def _prev_ids_of(seq: list[int]) -> torch.Tensor:
    return torch.tensor(
        [[seq[i - k] if i - k >= 0 else -1 for k in (1, 2)] for i in range(len(seq))]
    )


def test_full_history_matches_reference():
    m = _memory()
    seq = [3, 7, 7, 1, 9, 4]
    got = m.build_ngram_embeddings(torch.tensor(seq), _prev_ids_of(seq))
    torch.testing.assert_close(got, _reference_ngrams(m.embed, torch.tensor(seq)))


def test_window_sees_history_before_it():
    # A decode window over the last 3 tokens equals the same tokens computed
    # with the whole sequence: history comes from prev_ids, not the batch.
    m = _memory()
    seq = [3, 7, 7, 1, 9, 4]
    full = m.build_ngram_embeddings(torch.tensor(seq), _prev_ids_of(seq))
    window = m.build_ngram_embeddings(torch.tensor(seq[3:]), _prev_ids_of(seq)[3:])
    torch.testing.assert_close(window, full[3:])


def test_missing_history_is_zero():
    m = _memory()
    ids = torch.tensor([4, 2])
    got = m.build_ngram_embeddings(ids, torch.tensor([[-1, -1], [4, -1]]))
    e = m.embed(ids)
    torch.testing.assert_close(got[0, 0], e[0] / 2)
    torch.testing.assert_close(got[1, 1], (e[0] + e[1]) / 3)


@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available(),
    reason="needs a GPU or TRITON_INTERPRET=1",
)
def test_fill_prev_ids_kernel():
    device = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
    history = torch.full((4, 32), -1, dtype=torch.int32)
    history[2, :12] = torch.arange(100, 112)  # request in slot 2, 12 committed
    history[0, :3] = torch.tensor([50, 51, 52])  # request in slot 0, prefill
    # Batch: slot 2 decode window at positions 11..14 (last sampled + 3 drafts),
    # then slot 0 prefill chunk at positions 0..2, then 2 padding tokens.
    input_ids = torch.tensor([111, 7, 8, 9, 50, 51, 52, 0, 0], dtype=torch.int32)
    positions = torch.tensor([11, 12, 13, 14, 0, 1, 2, 0, 0])
    idx_mapping = torch.tensor([2, 0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4, 7], dtype=torch.int32)
    prev = torch.full((16, 2), -7, dtype=torch.int32)
    t = [x.to(device) for x in (prev, input_ids, positions, idx_mapping)]
    fill_ngm_prev_ids(
        t[0], t[1], t[2], t[3], query_start_loc.to(device), history.to(device)
    )
    expected = torch.tensor(
        [
            [110, 109],  # window start: both from history
            [111, 110],  # one from the batch, one from history
            [7, 111],
            [8, 7],
            [-1, -1],  # prefill at sequence start
            [50, -1],
            [51, 50],
        ],
        dtype=torch.int32,
    )
    got = t[0].cpu()
    torch.testing.assert_close(got[:7], expected)
    assert (got[7:] == -7).all()  # padding rows untouched
