import os

import torch

HEADS, V, K = 48, 128, 128
TILE = 32


def validate_mask(mask: torch.Tensor) -> None:
    if tuple(mask.shape) != (HEADS, K) or mask.dtype != torch.bool:
        raise ValueError(f"mask must be bool[{HEADS}, {K}]")
    if bool(mask.all()):
        raise ValueError("all-true mask is the off path")


def split_mask(mask_row: torch.Tensor):
    hi = mask_row.nonzero(as_tuple=False).flatten().to(torch.long)
    lo = (~mask_row).nonzero(as_tuple=False).flatten().to(torch.long)
    return hi, lo


def page_bytes(n_hi: int, n_lo: int) -> int:
    if n_hi + n_lo != K:
        raise ValueError("n_hi + n_lo must be 128")
    tiles = V // TILE
    return HEADS * V * n_hi * 2 + HEADS * V * n_lo * 1 + HEADS * tiles * 4


def pack_qk(x: torch.Tensor, hi_idx: torch.Tensor, lo_idx: torch.Tensor, n_v_heads: int):
    """Permute key channels into packed order, one copy per value head.

    ``x`` is [B, T, H, K]. Value heads that share a key head repeat that head,
    then gather fp16 channels before int8 channels. The update then loads two
    contiguous spans.
    """
    b, t, h, k = x.shape
    if n_v_heads % h != 0:
        raise ValueError("value heads must divide into key heads")
    group = n_v_heads // h
    flat = x.reshape(b * t, h, k).repeat_interleave(group, dim=1)
    order = torch.cat([hi_idx, lo_idx], dim=-1).to(device=x.device, dtype=torch.long)
    packed = torch.gather(flat, 2, order[None].expand(flat.shape[0], -1, -1))
    return packed.view(b, t, n_v_heads, k)


def store_head(plane: torch.Tensor, hi_idx: torch.Tensor, lo_idx: torch.Tensor):
    hi = plane[:, hi_idx].to(torch.float16).contiguous()
    tiles = V // TILE
    if lo_idx.numel() == 0:
        return hi, plane.new_empty(V, 0, dtype=torch.int8), plane.new_zeros(tiles)
    lo_f = plane[:, lo_idx].float().view(tiles, TILE, -1)
    amax = lo_f.abs().amax(dim=(1, 2)).clamp_min(1e-8)
    scale = amax / 127
    lo = (lo_f / scale[:, None, None]).round().clamp(-127, 127).to(torch.int8)
    return hi, lo.view(V, -1).contiguous(), scale


def reconstruct_head(hi, lo, scale, hi_idx, lo_idx):
    plane = hi.new_empty(V, K)
    if hi_idx.numel():
        plane[:, hi_idx] = hi.to(plane.dtype)
    if lo_idx.numel():
        row_scale = scale.to(plane.dtype).repeat_interleave(TILE)[:, None]
        plane[:, lo_idx] = lo.to(plane.dtype) * row_scale
    return plane


def update_head_plane(hi, lo, scale, hi_idx, lo_idx, fn):
    plane = reconstruct_head(hi, lo, scale, hi_idx, lo_idx)
    return store_head(fn(plane.float()).to(torch.float16), hi_idx, lo_idx)


def indexed_update(hi, lo, scale, indices, hi_idx, lo_idx, fn, scratch):
    """CPU reference. The server path must not call this.

    scratch is one (128, 128) buffer. No full-cache mirror.
    """
    if tuple(scratch.shape) != (V, K):
        raise ValueError("scratch is one head plane")
    for b in indices.tolist():
        for h in range(hi.shape[1]):
            row_hi = hi_idx[h] if hi_idx.ndim == 2 else hi_idx
            row_lo = lo_idx[h] if lo_idx.ndim == 2 else lo_idx
            plane = reconstruct_head(hi[b, h], lo[b, h], scale[b, h], row_hi, row_lo)
            scratch.copy_(fn(plane))
            h2, l2, s2 = store_head(scratch, row_hi, row_lo)
            hi[b, h] = h2
            lo[b, h] = l2
            scale[b, h] = s2


def rank_channels(error, decay, n_hi=32):
    # decay is the recorded keep factor. Larger means the head forgets slower.
    score = error * decay.clamp_min(0)[:, None]
    top = score.topk(n_hi, dim=1).indices
    mask = torch.zeros_like(error, dtype=torch.bool)
    mask.scatter_(1, top, True)
    return mask


def load_mask(path: str) -> torch.Tensor:
    mask = torch.load(path, weights_only=True)
    validate_mask(mask)
    counts = mask.sum(dim=1)
    if int(counts.min()) != int(counts.max()):
        raise ValueError("each head must keep the same n_hi")
    return mask


def indices_from_mask(mask: torch.Tensor):
    validate_mask(mask)
    counts = mask.sum(dim=1)
    if int(counts.min()) != int(counts.max()):
        raise ValueError("each head must keep the same n_hi")
    n_hi = int(counts[0])
    rows_hi, rows_lo = [], []
    for h in range(HEADS):
        hi, lo = split_mask(mask[h])
        rows_hi.append(hi)
        rows_lo.append(lo)
    hi_idx = torch.stack(rows_hi) if n_hi else torch.empty(HEADS, 0, dtype=torch.long)
    lo_idx = torch.stack(rows_lo)
    return hi_idx, lo_idx, n_hi


def enabled() -> bool:
    return os.environ.get("VLLM_DAMP_STATE", "off") == "mixed"


def temporal_page(num_v_heads: int, head_k_dim: int):
    """None when DAMP is off. Otherwise one uint8 temporal tensor, (nbytes,)."""
    if not enabled():
        return None
    path = os.environ.get("DAMP_MASK")
    if not path:
        raise RuntimeError("VLLM_DAMP_STATE=mixed requires DAMP_MASK")
    mask = load_mask(path)
    if tuple(mask.shape) != (num_v_heads, head_k_dim):
        raise RuntimeError(f"DAMP mask must be bool[{num_v_heads}, {head_k_dim}]")
    _hi, _lo, n_hi = indices_from_mask(mask)
    return (page_bytes(n_hi, head_k_dim - n_hi),), torch.uint8
