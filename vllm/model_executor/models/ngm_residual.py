# SPDX-License-Identifier: Apache-2.0
# N-gram residual (PioneerQyw/NGM math). Callable embed; 2D hidden OK.
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


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


def update_input_ids(cache: torch.Tensor | None, ids: torch.Tensor) -> torch.Tensor:
    if ids.dim() == 1:
        ids = ids.view(1, -1)
    if cache is None or ids.shape[-1] > 1:
        return ids
    return torch.cat([cache, ids], dim=-1)


class NgramResidual(nn.Module):
    """Cosine-gated 2-/3-gram residual. Same avg_pool / ReLU math as PioneerQyw."""

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

    def _lookup(self, input_ids: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(input_ids)
        if tokens.dim() == 2:
            tokens = tokens.view(*input_ids.shape, self.hidden_size)
        return tokens

    def build_ngram_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.view(1, -1)
        token_embeds = self._lookup(input_ids)
        B, L, _ = token_embeds.shape
        device = token_embeds.device
        all_ngram_embeds = []
        for n in self.ngram_sizes:
            if L < n:
                ngram_embed = torch.zeros(
                    B, L, self.hidden_size, device=device, dtype=token_embeds.dtype
                )
            else:
                padded = F.pad(token_embeds, (0, 0, n - 1, 0), mode="constant", value=0)
                padded_t = padded.transpose(1, 2)
                ngram_embed_t = F.avg_pool1d(
                    padded_t, kernel_size=n, stride=1, padding=0
                )
                ngram_embed = ngram_embed_t.transpose(1, 2)
            all_ngram_embeds.append(ngram_embed)
        return torch.stack(all_ngram_embeds, dim=2)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        squeezed = hidden_states.dim() == 2
        if squeezed:
            hidden_states = hidden_states.unsqueeze(0)
        if input_ids.dim() == 1:
            input_ids = input_ids.view(1, -1)
        _B, L, _D = hidden_states.shape
        ngram_embeds = self.build_ngram_embeddings(input_ids)[:, -L:, :, :].to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        hidden_norm = F.normalize(hidden_states, p=2, dim=-1)
        ngram_norm = F.normalize(ngram_embeds, p=2, dim=-1)
        similarity = torch.einsum("bld,blnd->bln", hidden_norm, ngram_norm)
        if self.use_relu:
            similarity = F.relu(similarity)
        update = (
            torch.einsum("bln,blnd->bld", similarity, ngram_embeds) * self.output_scale
        )
        out = hidden_states + update
        return out.squeeze(0) if squeezed else out
