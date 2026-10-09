"""Selective prefill using the native vLLM Llama/Qwen3 decoder layers.

This module owns request-local, pre-RoPE document caches. It does not implement
sampling or paged-cache allocation: the caller supplies a per-layer KV writer.
Timings synchronize the compute device at stage boundaries and include cache
transfers and the writer; they are diagnostic wall times, not kernel timings.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real
import time
from typing import Callable, Optional

import torch

from .attention import causal_attention
from .scoring import compute_attention_influence, fuse_scores, mean_attention_scores


_METHODS = {
    "prophet": 0.0,
    "prophet_fo": 0.0,
    "prophet_fo_residual": 1.0,
    "prophet_fo_mixed": 0.5,
}


@dataclass(frozen=True)
class RepairConfig:
    method: str = "prophet_fo"
    ratio: float = 0.2
    probe_layer: int = 2
    influence_weight: float = 0.5

    def __post_init__(self):
        if self.method not in _METHODS:
            raise ValueError(f"Unsupported repair method: {self.method!r}")
        for name in ("ratio", "influence_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        minimum = 0 if self.method == "prophet" else 1
        if (
            isinstance(self.probe_layer, bool)
            or not isinstance(self.probe_layer, Integral)
            or self.probe_layer < minimum
        ):
            raise ValueError(f"probe_layer must be an integer >= {minimum}")


@dataclass(frozen=True)
class _PendingRepair:
    prefix: tuple
    documents: tuple
    suffix: tuple
    config: RepairConfig

    @property
    def context_chunks(self):
        return tuple(chunk for chunk in (self.prefix, *self.documents) if chunk)

    @property
    def token_ids(self):
        return self.prefix + tuple(token for doc in self.documents for token in doc) + self.suffix


def _tokens(values, name):
    values = tuple(values)
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in values):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return tuple(int(x) for x in values)


def _tensor(result):
    return result[0] if isinstance(result, tuple) else result


class SparseRepairRuntime:
    def __init__(self, model, cache_device="cpu", attention_block_size=128):
        if (
            isinstance(attention_block_size, bool)
            or not isinstance(attention_block_size, Integral)
            or attention_block_size <= 0
        ):
            raise ValueError("attention_block_size must be a positive integer")
        self.model = model
        self.body = model.model
        self.layers = self.body.layers
        self.num_layers = len(self.layers)
        if self.num_layers == 0 or getattr(self.body, "start_layer", 0) != 0:
            raise ValueError("Repair runtime requires all decoder layers on one worker")
        if getattr(self.body, "end_layer", self.num_layers) != self.num_layers:
            raise ValueError("Pipeline-parallel decoder partitions are unsupported")
        self.device = self.body.embed_tokens.weight.device
        self.cache_device = torch.device(cache_device)
        self.attention_block_size = int(attention_block_size)
        self.cache = {}
        self.pending: Optional[_PendingRepair] = None
        self.metrics = {}

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _clock(self):
        self._sync()
        return time.perf_counter()

    @staticmethod
    def _pre_norm(layer, hidden, residual):
        if residual is None:
            residual = hidden
            hidden = _tensor(layer.input_layernorm(hidden))
        else:
            result = layer.input_layernorm(hidden, residual)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("Expected fused input RMSNorm to return hidden and residual")
            hidden, residual = result
        return hidden, residual

    @staticmethod
    def _project(layer, hidden):
        attn = layer.self_attn
        qkv = _tensor(attn.qkv_proj(hidden))
        q, k, v = qkv.split((attn.q_size, attn.kv_size, attn.kv_size), dim=-1)
        if hasattr(attn, "q_norm"):
            q = _tensor(attn.q_norm(q.reshape(-1, attn.num_heads, attn.head_dim))).reshape_as(q)
        if hasattr(attn, "k_norm"):
            k = _tensor(attn.k_norm(k.reshape(-1, attn.num_kv_heads, attn.head_dim))).reshape_as(k)
        return q, k, v

    @staticmethod
    def _rotate_pair(layer, positions, q, k):
        # vLLM rotary kernels may modify both arguments in place.
        return layer.self_attn.rotary_emb(positions, q.clone(), k.clone())

    @staticmethod
    def _rotate_keys(layer, positions, k):
        dummy = torch.zeros_like(k)
        _, rotated = layer.self_attn.rotary_emb(positions, dummy, k.clone())
        return rotated

    def _attention(self, layer, q, k, v, q_positions, key_positions):
        attn = layer.self_attn
        output = causal_attention(
            q.reshape(-1, attn.num_heads, attn.head_dim),
            k.reshape(-1, attn.num_kv_heads, attn.head_dim),
            v.reshape(-1, attn.num_kv_heads, attn.head_dim),
            q_positions, key_positions,
            query_block_size=self.attention_block_size,
        )
        return output.reshape(q.shape[0], attn.q_size)

    @staticmethod
    def _post_attention(layer, output, residual):
        output = _tensor(layer.self_attn.o_proj(output))
        normalized = layer.post_attention_layernorm(output, residual)
        if not isinstance(normalized, tuple) or len(normalized) != 2:
            raise TypeError("Expected fused post-attention RMSNorm to return hidden and residual")
        hidden, residual = normalized
        return _tensor(layer.mlp(hidden)), residual

    def _context(self, pending, layer_idx):
        entries = [self.cache[chunk][layer_idx] for chunk in pending.context_chunks]
        attn = self.layers[layer_idx].self_attn
        if not entries:
            weight = self.body.embed_tokens.weight
            empty = torch.empty((0, attn.kv_size), device=self.device, dtype=weight.dtype)
            return empty, empty.clone()
        keys = torch.cat([entry[0].to(self.device) for entry in entries], dim=0)
        values = torch.cat([entry[1].to(self.device) for entry in entries], dim=0)
        return keys, values

    @torch.no_grad()
    def prepare_chunks(self, chunks):
        """Cache independent chunks; retain only entries used by this request."""
        if self.pending is not None:
            raise RuntimeError("Disarm the pending request before preparing chunks")
        keys = list(dict.fromkeys(_tokens(chunk, "chunk") for chunk in chunks if len(chunk)))
        keep = set(keys)
        self.cache = {key: value for key, value in self.cache.items() if key in keep}
        start = self._clock()
        reused = len(self.cache)
        for key in keys:
            if key in self.cache:
                continue
            ids = torch.tensor(key, device=self.device, dtype=torch.long)
            positions = torch.arange(len(key), device=self.device, dtype=torch.long)
            hidden, residual = self.body.embed_tokens(ids), None
            entry = []
            for layer in self.layers:
                normalized, residual = self._pre_norm(layer, hidden, residual)
                q, k, v = self._project(layer, normalized)
                # clone also when the cache and compute device are identical.
                entry.append((k.detach().to(self.cache_device).clone(), v.detach().to(self.cache_device).clone()))
                q, k = self._rotate_pair(layer, positions, q, k)
                output = self._attention(layer, q, k, v, positions, positions)
                hidden, residual = self._post_attention(layer, output, residual)
            self.cache[key] = tuple(entry)
        elapsed = self._clock() - start
        return {
            "prepare_s": elapsed,
            "cached_chunks": len(self.cache),
            "new_chunks": len(self.cache) - reused,
            "reused_chunks": reused,
            "cached_tokens": sum(len(key) for key in self.cache),
            "cache_bytes": sum(t.numel() * t.element_size() for entry in self.cache.values() for pair in entry for t in pair),
        }

    def arm(self, prefix_ids, documents, suffix_ids, config: RepairConfig):
        if self.pending is not None:
            raise RuntimeError("A repair request is already pending")
        if not isinstance(config, RepairConfig):
            raise TypeError("config must be RepairConfig")
        if config.probe_layer >= self.num_layers:
            raise ValueError("probe_layer must be less than the decoder layer count")
        pending = _PendingRepair(
            _tokens(prefix_ids, "prefix_ids"),
            tuple(_tokens(doc, "document") for doc in documents),
            _tokens(suffix_ids, "suffix_ids"),
            config,
        )
        if not pending.suffix:
            raise ValueError("The query suffix must contain at least one token")
        if config.ratio < 1 and any(chunk not in self.cache for chunk in pending.context_chunks):
            raise ValueError("prepare_chunks must cache the prefix and all documents before arm")
        self.pending = pending

    def disarm(self):
        self.pending = None

    def clear(self):
        self.cache.clear()
        self.pending = None
        self.metrics = {}

    def _query_probe(self, pending, positions, doc_start, doc_end):
        ids = torch.tensor(pending.suffix, device=self.device, dtype=torch.long)
        hidden, residual = self.body.embed_tokens(ids), None
        query_positions = positions[doc_end:]
        scores = torch.zeros(doc_end - doc_start, device=self.device, dtype=torch.float32)
        for index, layer in enumerate(self.layers):
            normalized, residual = self._pre_norm(layer, hidden, residual)
            q, query_k, query_v = self._project(layer, normalized)
            context_k, context_v = self._context(pending, index)
            q, query_k = self._rotate_pair(layer, query_positions, q, query_k)
            context_k = self._rotate_keys(layer, positions[:doc_end], context_k)
            keys = torch.cat((context_k, query_k))
            values = torch.cat((context_v, query_v))
            attn = layer.self_attn
            layer_scores = mean_attention_scores(
                q.reshape(-1, attn.num_heads, attn.head_dim),
                keys.reshape(-1, attn.num_kv_heads, attn.head_dim),
                values.reshape(-1, attn.num_kv_heads, attn.head_dim),
                q_positions=query_positions,
                key_positions=positions,
            )
            scores += layer_scores[doc_start:doc_end].float() / self.num_layers
            output = self._attention(layer, q, keys, values, query_positions, positions)
            hidden, residual = self._post_attention(layer, output, residual)
        return scores

    def _fo_scores(self, pending, layer_idx, positions, q, k, v, doc_start, doc_end):
        layer = self.layers[layer_idx]
        attn = layer.self_attn
        context_k, context_v = self._context(pending, layer_idx)
        baseline_k = torch.cat((context_k, k[doc_end:]))
        baseline_v = torch.cat((context_v, v[doc_end:]))
        query, _ = self._rotate_pair(layer, positions[doc_end:], q[doc_end:], k[doc_end:])
        baseline_k = self._rotate_keys(layer, positions, baseline_k)
        new_k = self._rotate_keys(layer, positions[doc_start:doc_end], k[doc_start:doc_end])
        delta_k = new_k.float() - baseline_k[doc_start:doc_end].float()
        delta_v = v[doc_start:doc_end].float() - baseline_v[doc_start:doc_end].float()
        return compute_attention_influence(
            query.reshape(-1, attn.num_heads, attn.head_dim),
            baseline_k.reshape(-1, attn.num_kv_heads, attn.head_dim),
            baseline_v.reshape(-1, attn.num_kv_heads, attn.head_dim),
            delta_k.reshape(-1, attn.num_kv_heads, attn.head_dim),
            delta_v.reshape(-1, attn.num_kv_heads, attn.head_dim),
            target_start=doc_start,
            q_positions=positions[doc_end:],
            key_positions=positions,
            residual_weight=_METHODS[pending.config.method],
        )

    @staticmethod
    def _selected(scores, budget, doc_start, doc_end, count, device):
        if budget:
            chosen = scores.topk(budget).indices.add(doc_start).sort().values
        else:
            chosen = torch.empty(0, device=device, dtype=torch.long)
        return torch.cat((torch.arange(doc_start, device=device), chosen, torch.arange(doc_end, count, device=device)))

    @torch.no_grad()
    def run_prefill(self, input_ids, positions, write_kv: Callable):
        pending = self.pending
        if pending is None:
            raise RuntimeError("No repair request is armed")
        self.metrics = {}
        try:
            if input_ids.ndim != 1 or positions.shape != input_ids.shape:
                raise ValueError("Repair requires one flat full-prefill token and position sequence")
            if input_ids.dtype not in (torch.int32, torch.int64) or positions.dtype not in (torch.int32, torch.int64):
                raise ValueError("Token IDs and positions must use integer tensors")
            if input_ids.device != self.device or positions.device != self.device:
                raise ValueError("Tokens and positions must be on the model compute device")
            # Native vLLM 0.9.2 CUDA RoPE reads positions as int64_t, even
            # though the backend-independent attention helpers accept int32.
            positions = positions.to(dtype=torch.long)
            if tuple(input_ids.detach().cpu().tolist()) != pending.token_ids:
                raise ValueError("The engine prompt does not match the armed prefix/documents/suffix")
            if positions.numel() and (bool((positions < 0).any()) or bool((positions[1:] <= positions[:-1]).any())):
                raise ValueError("Positions must be nonnegative and strictly increasing")
            config = pending.config
            count = input_ids.numel()
            doc_start = len(pending.prefix)
            n_docs = sum(len(doc) for doc in pending.documents)
            doc_end = doc_start + n_docs
            budget = min(n_docs, max(1, int(n_docs * config.ratio))) if config.ratio > 0 else 0
            dense = config.ratio == 1.0 or n_docs == 0
            query_probe_s = fo_score_s = 0.0
            scores = torch.zeros(n_docs, device=self.device, dtype=torch.float32)
            if not dense and budget:
                start = self._clock()
                scores = self._query_probe(pending, positions, doc_start, doc_end)
                query_probe_s = self._clock() - start
            start = self._clock()
            active = torch.arange(count, device=self.device)
            if not dense and config.method == "prophet" and config.probe_layer == 0:
                active = self._selected(scores, budget, doc_start, doc_end, count, self.device)
            hidden, residual = self.body.embed_tokens(input_ids[active]), None
            for index, layer in enumerate(self.layers):
                normalized, residual = self._pre_norm(layer, hidden, residual)
                q, raw_k, v = self._project(layer, normalized)
                selecting = not dense and config.probe_layer > 0 and index == config.probe_layer
                if selecting:
                    if budget and config.method != "prophet":
                        score_start = self._clock()
                        influence = self._fo_scores(pending, index, positions, q, raw_k, v, doc_start, doc_end)
                        scores = fuse_scores(scores, influence, influence_weight=config.influence_weight)
                        fo_score_s = self._clock() - score_start
                    active = self._selected(scores, budget, doc_start, doc_end, count, self.device)
                    residual = residual[active]
                    # The probe layer has already projected every K/V row.
                    q, keys = self._rotate_pair(layer, positions, q, raw_k)
                    q = q[active]
                elif hidden.shape[0] == count:
                    q, keys = self._rotate_pair(layer, positions, q, raw_k)
                else:
                    context_k, context_v = self._context(pending, index)
                    suffix_zeros = raw_k.new_zeros((len(pending.suffix), raw_k.shape[-1]))
                    keys = torch.cat((context_k, suffix_zeros))
                    values = torch.cat((context_v, v.new_zeros(suffix_zeros.shape)))
                    keys[active], values[active] = raw_k, v
                    q, _ = self._rotate_pair(layer, positions[active], q, raw_k)
                    keys = self._rotate_keys(layer, positions, keys)
                    v = values
                attn = layer.self_attn
                write_kv(
                    index,
                    keys.reshape(count, attn.num_kv_heads, attn.head_dim).contiguous(),
                    v.reshape(count, attn.num_kv_heads, attn.head_dim).contiguous(),
                )
                output = self._attention(layer, q, keys, v, positions[active], positions)
                hidden, residual = self._post_attention(layer, output, residual)
            final = _tensor(self.body.norm(hidden, residual))
            output = final.new_zeros((count, final.shape[-1]))
            output[active] = final
            repair_s = self._clock() - start
            if dense:
                attention_work = kv_work = self.num_layers * n_docs
                budget = n_docs
            elif config.method == "prophet" and config.probe_layer == 0:
                attention_work = kv_work = self.num_layers * budget
            else:
                p = config.probe_layer
                attention_work = p * n_docs + (self.num_layers - p) * budget
                kv_work = (p + 1) * n_docs + (self.num_layers - p - 1) * budget
            denominator = self.num_layers * n_docs
            self.metrics = {
                "method": config.method, "ratio": config.ratio,
                "probe_layer": config.probe_layer,
                "influence_weight": config.influence_weight,
                "residual_weight": _METHODS[config.method],
                "num_tokens": count, "num_prefix_tokens": doc_start,
                "num_doc_tokens": n_docs, "num_suffix_tokens": len(pending.suffix),
                "selected_doc_tokens": budget,
                "selected_positions": active.detach().cpu().tolist(),
                "doc_attention_mlp_fraction": attention_work / denominator if denominator else 0.0,
                "doc_kv_projection_fraction": kv_work / denominator if denominator else 0.0,
                "query_probe_s": query_probe_s, "repair_s": repair_s,
                "fo_score_s": fo_score_s,
                "total_prefill_s": query_probe_s + repair_s,
                "timing_scope": "device_synchronized_wall_clock_including_cache_transfers_and_kv_writes",
                "influence_query_scope": "full_suffix",
            }
            return output
        finally:
            self.pending = None
