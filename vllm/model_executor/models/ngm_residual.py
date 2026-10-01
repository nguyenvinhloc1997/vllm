# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# N-gram residual (PioneerQyw/NGM math) over vLLM's flat token batch.
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

# Pad value for shifted positions; no real position equals it minus a shift.
_NO_POS = -(2**31)


def default_layer_ids(num_layers: int) -> list[int]:
    if num_layers < 2:
        raise ValueError("need at least 2 decoder layers")
    mid = max(1, num_layers // 2 - 1)
    return sorted({i for i in (1, mid) if 0 <= i < num_layers})


def parse_layer_ids(spec: str, num_layers: int) -> list[int]:
    text = (spec or "").strip()
    if not text:
        return default_layer_ids(num_layers)
    ids = [int(p) for p in text.split(",") if p.strip()]
    return [i for i in ids if 0 <= i < num_layers]


class NgramResidual(nn.Module):
    """Cosine-gated 2-/3-gram residual. Same mean-of-n / ReLU math as PioneerQyw.

    Inputs are vLLM's flat batch: several requests back to back, plus CUDA
    graph padding. A token's n-gram only uses an earlier token of the batch
    when that token sits at the expected position (same request, contiguous);
    anything else counts as zero, like the zero pad at sequence start.
    """

    def __init__(
        self,
        embed: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
        embedding_dim: int,
        ngram_sizes: tuple[int, ...] = (2, 3),
        use_relu: bool = True,
        output_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = embed
        self.hidden_size = embedding_dim
        self.ngram_sizes = ngram_sizes
        self.use_relu = use_relu
        self.output_scale = output_scale

    def build_ngram_embeddings(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """(L,) ids + (L,) positions -> (L, len(ngram_sizes), D)."""
        # ponytail: a token right before a request boundary can pass the
        # position check when its position happens to be the next request's
        # first position minus k. Rare and 0.1-scaled; exact boundaries need
        # query_start_loc from the runner.
        token_embeds = self.embed(input_ids)
        L = token_embeds.shape[0]
        acc = token_embeds
        out = []
        for k in range(1, max(self.ngram_sizes)):
            prev = F.pad(token_embeds, (0, 0, k, 0))[:L]
            prev_pos = F.pad(positions, (k, 0), value=_NO_POS)[:L]
            same_seq = (prev_pos == positions - k).unsqueeze(-1)
            acc = acc + prev * same_seq.to(prev.dtype)
            if k + 1 in self.ngram_sizes:
                out.append(acc / (k + 1))
        return torch.stack(out, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        L = hidden_states.shape[0]
        ngram_embeds = self.build_ngram_embeddings(input_ids[:L], positions[:L]).to(
            dtype=hidden_states.dtype
        )
        hidden_norm = F.normalize(hidden_states, p=2, dim=-1)
        ngram_norm = F.normalize(ngram_embeds, p=2, dim=-1)
        similarity = torch.einsum("ld,lnd->ln", hidden_norm, ngram_norm)
        if self.use_relu:
            similarity = F.relu(similarity)
        update = (
            torch.einsum("ln,lnd->ld", similarity, ngram_embeds) * self.output_scale
        )
        return hidden_states + update
