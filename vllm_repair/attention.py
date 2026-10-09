"""Causal prefill attention for queries at explicit, possibly sparse positions."""

from numbers import Integral

import torch
import torch.nn.functional as F


def _validate(q, k, v, q_positions, key_positions, query_block_size):
    if (
        isinstance(query_block_size, bool)
        or not isinstance(query_block_size, Integral)
        or query_block_size <= 0
    ):
        raise ValueError("query_block_size must be a positive integer")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(f"{name} must be a rank-3 tensor")
        if tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise ValueError(f"{name} must have a supported floating-point dtype")
        if tensor.device != q.device or tensor.dtype != q.dtype:
            raise ValueError("q, k and v must have the same device and dtype")
    n_query, n_heads, head_dim = q.shape
    n_keys, n_kv_heads, key_dim = k.shape
    if min(n_heads, n_kv_heads, head_dim) <= 0:
        raise ValueError("head counts and head dimensions must be positive")
    if n_heads % n_kv_heads:
        raise ValueError("the query head count must be divisible by the KV head count")
    if key_dim != head_dim or v.shape != (n_keys, n_kv_heads, head_dim):
        raise ValueError("q, k and v have incompatible token or head dimensions")
    for name, positions, length in (
        ("q_positions", q_positions, n_query),
        ("key_positions", key_positions, n_keys),
    ):
        if not isinstance(positions, torch.Tensor) or positions.shape != (length,):
            raise ValueError(f"{name} must be a vector of length {length}")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must have dtype int32 or int64")
        if positions.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")


@torch.no_grad()
def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    query_block_size: int = 128,
) -> torch.Tensor:
    """Compute causal attention using absolute positions instead of row indices.

    Shapes are q=[Q,Hq,D], k/v=[N,Hkv,D], with Hq divisible by Hkv.
    Inputs must already have their positional/RoPE transformation applied.
    Keys are visible iff key_positions <= q_positions; positions may be sparse
    or unordered. Consecutive groups of Hq/Hkv query heads share a KV head.

    Uses PyTorch SDPA with no dropout and its default 1/sqrt(D) scale. The
    explicit boolean mask is essential: ``is_causal=True`` would align token
    indices rather than the supplied positions. Only a query-block-by-N mask
    is constructed. The selected SDPA backend controls attention workspace.

    Returns [Q,Hq,D] on the original device and with the original dtype,
    including under autocast. Empty queries return an empty tensor; queries
    without a visible key return zero. This inference helper keeps no graph.
    """
    _validate(q, k, v, q_positions, key_positions, query_block_size)
    output = torch.empty_like(q)
    if q.shape[0] == 0:
        return output
    if k.shape[0] == 0:
        return output.zero_()

    grouped = q.shape[1] != k.shape[1]
    keys = k.transpose(0, 1).unsqueeze(0)
    values = v.transpose(0, 1).unsqueeze(0)
    with torch.autocast(device_type=q.device.type, enabled=False):
        for start in range(0, q.shape[0], query_block_size):
            end = min(start + query_block_size, q.shape[0])
            allowed = key_positions[None, :] <= q_positions[start:end, None]
            attended = F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0),
                keys,
                values,
                attn_mask=allowed[None, None, :, :],
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=grouped,
            )
            # Explicitly define fully masked rows, independent of SDPA backend.
            attended = attended.masked_fill(
                ~allowed.any(dim=-1)[None, None, :, None], 0.0
            )
            output[start:end] = attended.squeeze(0).transpose(0, 1)
    return output
