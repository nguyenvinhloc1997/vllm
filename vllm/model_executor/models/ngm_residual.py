# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# N-gram residual (PioneerQyw/NGM math) over vLLM's flat token batch.
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.triton_utils import tl, triton

# History length: a 3-gram needs the two tokens before each token.
NGM_HISTORY = 2


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


@triton.jit
def _ngm_prev_ids_kernel(
    prev_ids_ptr,
    input_ids_ptr,
    positions_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    HISTORY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    for start in range(query_start, query_end, BLOCK):
        i = start + tl.arange(0, BLOCK)
        valid = (i < query_end) & (req_state_idx >= 0)
        pos = tl.load(positions_ptr + i, mask=valid, other=0)
        for k in tl.static_range(1, HISTORY + 1):
            # Inside this request's window: the batch holds the token (drafts
            # included). Before it: committed history in all_token_ids.
            in_window = i - k >= query_start
            from_batch = tl.load(
                input_ids_ptr + i - k, mask=valid & in_window, other=-1
            )
            in_history = valid & ~in_window & (pos - k >= 0)
            from_history = tl.load(
                all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos - k,
                mask=in_history,
                other=-1,
            )
            prev = tl.where(in_window, from_batch, from_history)
            tl.store(prev_ids_ptr + i * HISTORY + (k - 1), prev, mask=valid)


def fill_ngm_prev_ids(
    # [max_num_tokens, NGM_HISTORY]; -1 marks "no such token"
    prev_ids: torch.Tensor,
    # [max_num_tokens]
    input_ids: torch.Tensor,
    # [max_num_tokens], sequence index of each token
    positions: torch.Tensor,
    # [num_reqs] batch_idx -> req_state_idx
    idx_mapping: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
    # [max_num_reqs, max_model_len], committed tokens per request
    all_token_ids: torch.Tensor,
) -> None:
    """Write each scheduled token's previous NGM_HISTORY token ids."""
    num_reqs = idx_mapping.shape[0]
    _ngm_prev_ids_kernel[(num_reqs,)](
        prev_ids,
        input_ids,
        positions,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        HISTORY=NGM_HISTORY,
        BLOCK=128,
    )


class NgramResidual(nn.Module):
    """Cosine-gated 2-/3-gram residual. Same mean-of-n / ReLU math as PioneerQyw.

    `prev_ids[:, k - 1]` is the id k tokens before each token in its own request,
    or -1 at sequence start; a missing token counts as a zero embedding, like
    the original zero pad.
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
        assert max(ngram_sizes) <= NGM_HISTORY + 1
        self.embed = embed
        self.hidden_size = embedding_dim
        self.ngram_sizes = ngram_sizes
        self.use_relu = use_relu
        self.output_scale = output_scale

    def build_ngram_embeddings(
        self, input_ids: torch.Tensor, prev_ids: torch.Tensor
    ) -> torch.Tensor:
        """(L,) ids + (L, NGM_HISTORY) previous ids -> (L, len(ngram_sizes), D)."""
        acc = self.embed(input_ids)
        out = []
        for k in range(1, max(self.ngram_sizes)):
            prev = prev_ids[:, k - 1]
            present = (prev >= 0).unsqueeze(-1)
            acc = acc + self.embed(prev.clamp(min=0)) * present.to(acc.dtype)
            if k + 1 in self.ngram_sizes:
                out.append(acc / (k + 1))
        return torch.stack(out, dim=1)

    def ngrams(
        self, input_ids: torch.Tensor, prev_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """N-gram embeddings and their unit vectors, shared by every NGM layer."""
        ngram_embeds = self.build_ngram_embeddings(input_ids, prev_ids)
        return ngram_embeds, F.normalize(ngram_embeds, p=2, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        ngrams: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        ngram_embeds, ngram_norm = ngrams
        hidden_norm = F.normalize(hidden_states, p=2, dim=-1)
        similarity = torch.einsum(
            "ld,lnd->ln", hidden_norm, ngram_norm.to(hidden_states.dtype)
        )
        if self.use_relu:
            similarity = F.relu(similarity)
        update = (
            torch.einsum("ln,lnd->ld", similarity, ngram_embeds.to(hidden_states.dtype))
            * self.output_scale
        )
        return hidden_states + update
