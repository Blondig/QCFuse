"""Query-conditioned, first-order scores for selective KV recomputation.

Keys and queries must already use the same positional/RoPE convention. Scores
measure a local attention-output change with the query held fixed; they are not
an estimate of the full network's final output error.
"""

import math
from numbers import Integral, Real

import torch


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _weight(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def _finite(tensors, message: str) -> None:
    # One host synchronization for the whole check, not one per tensor/block.
    if not bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all()):
        raise ValueError(message)


def _validate_inputs(q, k, v, delta_k, delta_v, q_positions, key_positions):
    tensors = (q, k, v, delta_k, delta_v)
    for name, tensor in zip(("q", "k", "v", "delta_k", "delta_v"), tensors):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(f"{name} must be a rank-3 tensor")
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must have a floating-point dtype")
        if tensor.device != q.device:
            raise ValueError("q, k, v and both deltas must use the same device")

    n_query, n_heads, head_dim = q.shape
    n_keys, n_kv_heads, key_dim = k.shape
    n_targets = delta_k.shape[0]
    if min(n_heads, n_kv_heads, head_dim, v.shape[2]) <= 0:
        raise ValueError("head counts and head dimensions must be positive")
    if n_heads % n_kv_heads:
        raise ValueError("the query head count must be divisible by the KV head count")
    if key_dim != head_dim or v.shape[:2] != (n_keys, n_kv_heads):
        raise ValueError("q, k and v have incompatible head or token dimensions")
    if delta_k.shape != (n_targets, n_kv_heads, head_dim):
        raise ValueError("delta_k must match k's head dimensions")
    if delta_v.shape != (n_targets, n_kv_heads, v.shape[2]):
        raise ValueError("delta_v must match delta_k's token count and v's heads")
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
    _finite(tensors, "q, k, v and both deltas must contain only finite values")


def _principal_direction(delta_k, delta_v, power_iterations, max_pca_tokens):
    """Approximate the leading uncentered drift direction, without a full SVD."""
    n_tokens = delta_k.shape[0]
    sample_count = min(n_tokens, max_pca_tokens)
    if sample_count == 1:
        indices = torch.tensor([n_tokens // 2], device=delta_k.device)
    else:
        indices = torch.div(
            torch.arange(sample_count, device=delta_k.device) * (n_tokens - 1),
            sample_count - 1,
            rounding_mode="floor",
        )
    sample = torch.cat(
        (
            delta_k.index_select(0, indices).float().flatten(1),
            delta_v.index_select(0, indices).float().flatten(1),
        ),
        dim=1,
    )
    tiny = torch.finfo(torch.float32).tiny
    # A common scale preserves the direction and limits overflow in X^T X.
    sample = sample / sample.abs().amax().clamp_min(tiny)
    row_energy = sample.square().sum(dim=1)
    largest_row = sample.index_select(0, row_energy.argmax().reshape(1)).squeeze(0)
    dense_start = torch.cos(
        torch.arange(sample.shape[1], device=sample.device, dtype=torch.float32)
        + 0.5
    )
    # Two deterministic starts reduce the chance of missing an eigenspace.
    directions = torch.stack((largest_row, dense_start), dim=1)
    directions = directions / directions.norm(dim=0).clamp_min(tiny)
    for _ in range(power_iterations):
        directions = sample.T @ (sample @ directions)
        directions = directions / directions.norm(dim=0).clamp_min(tiny)
    energy = (sample @ directions).square().sum(dim=0)
    return directions.index_select(1, energy.argmax().reshape(1)).squeeze(1)


def _logits(q_grouped, keys, q_positions, key_positions, group_size):
    logits = torch.bmm(q_grouped, keys.float().permute(1, 2, 0))
    n_query = q_positions.numel()
    allowed = key_positions[None, :] <= q_positions[:, None]
    return logits.view(
        logits.shape[0], group_size, n_query, keys.shape[0]
    ).masked_fill(~allowed[None, None, :, :], -torch.inf).flatten(1, 2)


def _baseline(
    q_grouped, k, v, q_positions, key_positions, group_size, block_size,
    *, compute_output=True,
):
    """Stream the full causal denominator and output for one query block."""
    shape = q_grouped.shape[:2]
    running_max = torch.full(shape, -torch.inf, device=k.device, dtype=torch.float32)
    denominator = torch.zeros(shape, device=k.device, dtype=torch.float32)
    numerator = None
    if compute_output:
        numerator = torch.zeros(
            (*shape, v.shape[-1]), device=k.device, dtype=torch.float32
        )
    for start in range(0, k.shape[0], block_size):
        end = min(start + block_size, k.shape[0])
        logits = _logits(
            q_grouped, k[start:end], q_positions, key_positions[start:end], group_size
        )
        new_max = torch.maximum(running_max, logits.amax(dim=-1))
        safe_max = torch.where(torch.isfinite(new_max), new_max, 0.0)
        old_scale = torch.exp(running_max - safe_max)
        probabilities = torch.exp(logits - safe_max.unsqueeze(-1))
        if compute_output:
            numerator = numerator * old_scale.unsqueeze(-1) + torch.bmm(
                probabilities, v[start:end].float().permute(1, 0, 2)
            )
        denominator = denominator * old_scale + probabilities.sum(dim=-1)
        running_max = new_max
    # Empty causal rows have zero output and zero attention probabilities.
    output = None
    if compute_output:
        output = numerator / denominator.clamp_min(1.0).unsqueeze(-1)
    lse = torch.where(
        denominator > 0,
        running_max + denominator.clamp_min(1.0).log(),
        torch.full_like(running_max, torch.inf),
    )
    return output, lse


def _influence_norm(q_grouped, output, values, delta_k, delta_v):
    """Expand the squared norm without a query/token/head/value tensor."""
    values = values.permute(1, 0, 2)
    delta_v = delta_v.permute(1, 0, 2)
    delta_logits = torch.bmm(q_grouped, delta_k.permute(1, 2, 0))
    dv_squared = delta_v.square().sum(dim=-1).unsqueeze(1)
    cross = (delta_v * values).sum(dim=-1).unsqueeze(1) - torch.bmm(
        output, delta_v.transpose(1, 2)
    )
    # Direct distance accumulation avoids catastrophic cancellation in
    # ||v||^2 + ||o||^2 - 2<v,o>. cdist does not allocate a [H,Q,N,Dv] tensor.
    centered_value_squared = torch.cdist(
        output, values, p=2.0, compute_mode="donot_use_mm_for_euclid_dist"
    ).square()
    value_scale_squared = values.square().sum(dim=-1).unsqueeze(1) + (
        output.square().sum(dim=-1).unsqueeze(-1)
    )
    is_roundoff = centered_value_squared <= (
        64.0 * torch.finfo(torch.float32).eps**2 * value_scale_squared
    )
    centered_value_squared = centered_value_squared.masked_fill(is_roundoff, 0.0)
    cross = cross.masked_fill(is_roundoff, 0.0)
    return (
        dv_squared
        + 2.0 * delta_logits * cross
        + delta_logits.square() * centered_value_squared
    ).clamp_min(0.0)


def _discard_projection_roundoff(delta_k, delta_v, residual_k, residual_v):
    """Discard projection noise relative to each token's original drift norm."""
    scale = torch.maximum(
        delta_k.abs().amax(dim=(1, 2)), delta_v.abs().amax(dim=(1, 2))
    )
    # Shared per-token scaling avoids an absolute cutoff and energy underflow.
    scale = torch.where(scale > 0, scale, 1.0)[:, None, None]
    raw_energy = (delta_k / scale).square().sum(dim=(1, 2)) + (
        delta_v / scale
    ).square().sum(dim=(1, 2))
    residual_energy = (residual_k / scale).square().sum(dim=(1, 2)) + (
        residual_v / scale
    ).square().sum(dim=(1, 2))
    tolerance = 64.0 * torch.finfo(torch.float32).eps**2
    is_roundoff = (residual_energy <= tolerance * raw_energy)[:, None, None]
    return residual_k.masked_fill(is_roundoff, 0.0), residual_v.masked_fill(
        is_roundoff, 0.0
    )


@torch.no_grad()
def compute_attention_influence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    delta_k: torch.Tensor,
    delta_v: torch.Tensor,
    *,
    target_start: int,
    q_positions: torch.Tensor,
    key_positions: torch.Tensor,
    residual_weight: float = 0.0,
    key_block_size: int = 256,
    query_block_size: int = 16,
    power_iterations: int = 8,
    max_pca_tokens: int = 2048,
) -> torch.Tensor:
    """Return float32 FO scores for a contiguous target range in ``k``/``v``.

    Shapes are q=[Q,Hq,D], k=[N,Hkv,D], v=[N,Hkv,Dv],
    delta_k=[T,Hkv,D], and delta_v=[T,Hkv,Dv]. Each score averages over
    the Q query rows and sums the squared FO output change over query heads.
    GQA maps each consecutive group of Hq/Hkv query heads to one KV head.

    The baseline softmax includes every causally visible key, including keys
    outside the target range. Rows without visible keys contribute zero. Empty
    query or target tensors produce zero scores. Positions need not be sorted.

    ``residual_weight`` linearly mixes raw scores and scores after removing
    one uncentered, joint K/V drift direction across the target tokens. This
    optional operation can remove useful shared value changes; zero is the
    default. The direction uses deterministic sampling and power iteration.
    Projection residuals within eight float32 epsilons of the original drift
    norm are zeroed before scoring, so rank fusion cannot amplify roundoff.

    All scoring arithmetic uses float32, with autocast disabled. Matmul still
    respects the caller's backend precision settings. No gradients are kept.
    Value-output distances use a direct distance kernel to avoid subtracting
    large squared norms. Distances within float32 rounding scale are zeroed;
    genuine variations at that scale cannot be reliably distinguished. The
    direct distance kernel may be slower than GEMM on GPU.
    """
    residual_weight = _weight("residual_weight", residual_weight)
    for name, value in (
        ("key_block_size", key_block_size),
        ("query_block_size", query_block_size),
        ("power_iterations", power_iterations),
        ("max_pca_tokens", max_pca_tokens),
    ):
        _positive_int(name, value)
    _validate_inputs(q, k, v, delta_k, delta_v, q_positions, key_positions)
    if (
        isinstance(target_start, bool)
        or not isinstance(target_start, Integral)
        or target_start < 0
        or target_start + delta_k.shape[0] > k.shape[0]
    ):
        raise ValueError("target_start and the delta length must define a range within k")

    n_query, n_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    n_targets = delta_k.shape[0]
    group_size = n_heads // n_kv_heads
    scores = torch.zeros(n_targets, device=q.device, dtype=torch.float32)
    if n_query == 0 or n_targets == 0:
        return scores

    with torch.autocast(device_type=q.device.type, enabled=False):
        direction = None
        if residual_weight > 0:
            direction = _principal_direction(
                delta_k, delta_v, power_iterations, max_pca_tokens
            )
            key_direction = direction[: n_kv_heads * head_dim].view(n_kv_heads, head_dim)
            value_direction = direction[n_kv_heads * head_dim :].view(
                n_kv_heads, v.shape[-1]
            )
        for query_start in range(0, n_query, query_block_size):
            query_end = min(query_start + query_block_size, n_query)
            query_count = query_end - query_start
            positions = q_positions[query_start:query_end]
            q_grouped = (
                q[query_start:query_end]
                .float()
                .reshape(query_count, n_kv_heads, group_size, head_dim)
                .permute(1, 2, 0, 3)
                .reshape(n_kv_heads, group_size * query_count, head_dim)
                / math.sqrt(head_dim)
            )
            output, lse = _baseline(
                q_grouped, k, v, positions, key_positions, group_size, key_block_size
            )
            for start in range(0, n_targets, key_block_size):
                end = min(start + key_block_size, n_targets)
                key_start, key_end = target_start + start, target_start + end
                logits = _logits(
                    q_grouped,
                    k[key_start:key_end],
                    positions,
                    key_positions[key_start:key_end],
                    group_size,
                )
                attention_squared = torch.exp(logits - lse.unsqueeze(-1)).square()
                values = v[key_start:key_end].float()
                dk = delta_k[start:end].float()
                dv = delta_v[start:end].float()
                norm = torch.zeros_like(attention_squared)
                if residual_weight < 1.0:
                    norm = (1.0 - residual_weight) * _influence_norm(
                        q_grouped, output, values, dk, dv
                    )
                if direction is not None:
                    projection = (dk * key_direction).sum(dim=(1, 2)) + (
                        dv * value_direction
                    ).sum(dim=(1, 2))
                    residual_k = dk - projection[:, None, None] * key_direction
                    residual_v = dv - projection[:, None, None] * value_direction
                    residual_k, residual_v = _discard_projection_roundoff(
                        dk, dv, residual_k, residual_v
                    )
                    norm = norm + residual_weight * _influence_norm(
                        q_grouped, output, values, residual_k, residual_v
                    )
                scores[start:end] += (attention_squared * norm).sum(dim=(0, 1)) / n_query
    _finite((scores,), "attention influence overflowed float32; check input magnitudes")
    return scores


@torch.no_grad()
def mean_attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_positions: torch.Tensor,
    key_positions: torch.Tensor,
    key_block_size: int = 256,
    query_block_size: int = 16,
) -> torch.Tensor:
    """Return attention probabilities averaged over query rows and query heads.

    Shapes are q=[Q,Hq,D], k=[N,Hkv,D], and v=[N,Hkv,Dv]. The result
    is float32[N]. All causally visible keys participate in each denominator;
    no candidate restriction or renormalization is applied. Empty query sets
    and rows without visible keys contribute zero. Each nonempty query/head
    row contributes mass 1/(Q*Hq), so the scores sum to one if all rows have
    at least one visible key.

    The computation streams over query and key blocks in float32. Values are
    validated for API consistency but do not enter the attention probabilities.
    This helper can be summed across layers for a query-relevance selector.
    """
    _positive_int("key_block_size", key_block_size)
    _positive_int("query_block_size", query_block_size)
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(f"{name} must be a rank-3 tensor")
    _validate_inputs(q, k, v, k[:0], v[:0], q_positions, key_positions)

    n_query, n_heads, head_dim = q.shape
    n_keys, n_kv_heads, _ = k.shape
    scores = torch.zeros(n_keys, device=q.device, dtype=torch.float32)
    if n_query == 0 or n_keys == 0:
        return scores
    group_size = n_heads // n_kv_heads
    with torch.autocast(device_type=q.device.type, enabled=False):
        for query_start in range(0, n_query, query_block_size):
            query_end = min(query_start + query_block_size, n_query)
            query_count = query_end - query_start
            positions = q_positions[query_start:query_end]
            q_grouped = (
                q[query_start:query_end]
                .float()
                .reshape(query_count, n_kv_heads, group_size, head_dim)
                .permute(1, 2, 0, 3)
                .reshape(n_kv_heads, group_size * query_count, head_dim)
                / math.sqrt(head_dim)
            )
            _, lse = _baseline(
                q_grouped,
                k,
                v,
                positions,
                key_positions,
                group_size,
                key_block_size,
                compute_output=False,
            )
            for start in range(0, n_keys, key_block_size):
                end = min(start + key_block_size, n_keys)
                logits = _logits(
                    q_grouped,
                    k[start:end],
                    positions,
                    key_positions[start:end],
                    group_size,
                )
                probabilities = torch.exp(logits - lse.unsqueeze(-1))
                scores[start:end] += probabilities.sum(dim=(0, 1)) / (n_query * n_heads)
    _finite((scores,), "attention probabilities overflowed float32; check input magnitudes")
    return scores


def _validate_score_vectors(*vectors):
    for scores in vectors:
        if not isinstance(scores, torch.Tensor) or scores.ndim != 1:
            raise ValueError("scores must be one-dimensional tensors")
        if not scores.is_floating_point():
            raise ValueError("scores must have floating-point dtypes")
    if any(scores.device != vectors[0].device for scores in vectors):
        raise ValueError("score vectors must use the same device")
    _finite(vectors, "scores must contain only finite values")


def _percentile_rank(scores):
    count = scores.numel()
    if count <= 1:
        return torch.zeros_like(scores, dtype=torch.float32)
    sorted_scores, order = scores.sort()
    indices = torch.arange(count, device=scores.device)
    start = torch.cat(
        (
            torch.ones(1, device=scores.device, dtype=torch.bool),
            sorted_scores[1:] != sorted_scores[:-1],
        )
    )
    end = torch.cat((start[1:], torch.ones(1, device=scores.device, dtype=torch.bool)))
    first = torch.where(start, indices, 0).cummax(dim=0).values
    last = torch.where(end, indices, count - 1).flip(0).cummin(dim=0).values.flip(0)
    ranks = (first.float() + last.float()) / (2.0 * (count - 1))
    ranks = torch.where(sorted_scores[-1] > sorted_scores[0], ranks, 0.0)
    result = torch.empty(count, device=scores.device, dtype=torch.float32)
    result.scatter_(0, order, ranks)
    return result


@torch.no_grad()
def percentile_rank(scores: torch.Tensor) -> torch.Tensor:
    """Return average zero-based ranks / (N-1); constant vectors return zero.

    Ties receive their group's mean rank. Therefore a tied endpoint need not
    receive exactly zero or one. Empty and singleton vectors also return zero.
    """
    _validate_score_vectors(scores)
    return _percentile_rank(scores)


@torch.no_grad()
def fuse_scores(
    relevance: torch.Tensor,
    influence: torch.Tensor,
    influence_weight: float = 0.5,
) -> torch.Tensor:
    """Mix tie-aware relevance and influence percentiles in float32."""
    influence_weight = _weight("influence_weight", influence_weight)
    _validate_score_vectors(relevance, influence)
    if relevance.shape != influence.shape:
        raise ValueError("relevance and influence must have the same shape")
    return (1.0 - influence_weight) * _percentile_rank(
        relevance
    ) + influence_weight * _percentile_rank(influence)
