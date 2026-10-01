# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F

from vllm.model_executor.models.ngm_residual import NgramResidual

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


def test_single_sequence_matches_reference():
    m = _memory()
    ids = torch.tensor([3, 7, 7, 1, 9, 4])
    got = m.build_ngram_embeddings(ids, torch.arange(6))
    torch.testing.assert_close(got, _reference_ngrams(m.embed, ids))


def test_requests_do_not_mix():
    m = _memory()
    ids_a, pos_a = torch.tensor([3, 7, 7, 1, 9]), torch.arange(5)
    # A decode window: starts mid-sequence, no history in the batch.
    ids_b, pos_b = torch.tensor([4, 2, 8, 5]), torch.arange(40, 44)
    pad_ids, pad_pos = torch.zeros(3, dtype=torch.long), torch.zeros(3)
    ids = torch.cat([ids_a, ids_b, pad_ids])
    pos = torch.cat([pos_a, pos_b, pad_pos.long()])
    hidden = torch.randn(len(ids), D)

    batched = m(hidden, ids, pos)
    alone_a = m(hidden[:5], ids_a, pos_a)
    alone_b = m(hidden[5:9], ids_b, pos_b)
    torch.testing.assert_close(batched[:5], alone_a)
    torch.testing.assert_close(batched[5:9], alone_b)


def test_window_start_has_no_history():
    m = _memory()
    ids = torch.tensor([4, 2, 8])
    got = m.build_ngram_embeddings(ids, torch.arange(40, 43))
    e = m.embed(ids)
    torch.testing.assert_close(got[0, 0], e[0] / 2)
    torch.testing.assert_close(got[0, 1], e[0] / 3)
    torch.testing.assert_close(got[2, 1], e.sum(0) / 3)
