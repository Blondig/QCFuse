"""Independent CPU references for the vLLM repair attention and scores."""

import importlib.util
import math
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"repair_{name}", ROOT / "vllm_repair" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attention = _load("attention")
scoring = _load("scoring")


def _dense(q, k, v, q_positions, key_positions):
    """Float64 reference with explicit KV-head expansion and causal positions."""
    repeat = q.shape[1] // k.shape[1]
    q = q.double()
    k = k.double().repeat_interleave(repeat, dim=1)
    v = v.double().repeat_interleave(repeat, dim=1)
    allowed = key_positions[None, :] <= q_positions[:, None]
    logits = torch.einsum("qhd,nhd->qhn", q, k) / math.sqrt(q.shape[-1])
    weights = logits.masked_fill(~allowed[:, None, :], -torch.inf).softmax(-1)
    weights = torch.where(allowed.any(-1)[:, None, None], weights, 0.0)
    return weights, torch.einsum("qhn,nhv->qhv", weights, v)


def _case(query_heads=4, kv_heads=2, value_dim=5):
    generator = torch.Generator().manual_seed(73)
    q = torch.randn(5, query_heads, 5, generator=generator)
    k = torch.randn(9, kv_heads, 5, generator=generator)
    v = torch.randn(9, kv_heads, value_dim, generator=generator)
    # Both orders are deliberately nonmonotonic, and one row sees no key.
    q_positions = torch.tensor([12, -1, 3, 20, 9])
    key_positions = torch.tensor([13, 0, 8, 4, 1, 21, 3, 7, 11])
    return q, k, v, q_positions, key_positions


class StandaloneRepairMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_sdpa_sparse_unordered_positions_and_grouped_heads(self):
        for query_heads, kv_heads in ((4, 2), (4, 1), (2, 2)):
            q, k, v, qp, kp = _case(query_heads, kv_heads)
            _, expected = _dense(q, k, v, qp, kp)
            for block in (1, 3, 128):
                with self.subTest(heads=(query_heads, kv_heads), block=block):
                    actual = attention.causal_attention(
                        q, k, v, qp, kp, query_block_size=block
                    )
                    torch.testing.assert_close(
                        actual.double(), expected, rtol=2e-5, atol=2e-7
                    )
                    self.assertEqual(torch.count_nonzero(actual[1]).item(), 0)

    def test_sdpa_dtype_autocast_and_empty_boundaries(self):
        q, k, v, qp, kp = _case()
        for dtype, tolerance in (
            (torch.float16, 2e-3),
            (torch.bfloat16, 1e-2),
            (torch.float64, 1e-12),
        ):
            with self.subTest(dtype=dtype):
                operands = [tensor.to(dtype) for tensor in (q, k, v)]
                _, expected = _dense(*operands, qp, kp)
                actual = attention.causal_attention(*operands, qp, kp)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(
                    actual.double(), expected, rtol=tolerance, atol=tolerance
                )
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = attention.causal_attention(q, k, v, qp, kp)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, attention.causal_attention(q, k, v, qp, kp))
        self.assertEqual(
            attention.causal_attention(q[:0], k, v, qp[:0], kp).shape,
            q[:0].shape,
        )
        torch.testing.assert_close(
            attention.causal_attention(q, k[:0], v[:0], qp, kp[:0]),
            torch.zeros_like(q),
        )

    def test_mean_scores_use_full_denominator_and_average_query_heads(self):
        q = torch.zeros(2, 4, 2)
        k = torch.zeros(5, 2, 2)
        v = torch.zeros(5, 2, 3)
        actual = scoring.mean_attention_scores(
            q,
            k,
            v,
            q_positions=torch.tensor([1, 4]),
            key_positions=torch.arange(5),
            key_block_size=2,
            query_block_size=1,
        )
        # First row attends to two keys, the second to all five.
        torch.testing.assert_close(actual, torch.tensor([0.35, 0.35, 0.1, 0.1, 0.1]))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertAlmostEqual(actual.sum().item(), 1.0, places=6)

    def test_mean_scores_match_dense_with_masked_rows_and_chunking(self):
        q, k, v, qp, kp = _case(value_dim=3)
        expected, _ = _dense(q, k, v, qp, kp)
        for query_block, key_block in ((1, 1), (2, 4), (16, 256)):
            actual = scoring.mean_attention_scores(
                q,
                k,
                v,
                q_positions=qp,
                key_positions=kp,
                query_block_size=query_block,
                key_block_size=key_block,
            )
            torch.testing.assert_close(
                actual.double(), expected.mean((0, 1)), rtol=2e-5, atol=1e-7
            )
            self.assertAlmostEqual(actual.sum().item(), 0.8, places=6)
        empty_queries = scoring.mean_attention_scores(
            q[:0], k, v, q_positions=qp[:0], key_positions=kp
        )
        torch.testing.assert_close(empty_queries, torch.zeros(len(k)))
        self.assertEqual(
            scoring.mean_attention_scores(
                q, k[:0], v[:0], q_positions=qp, key_positions=kp[:0]
            ).numel(),
            0,
        )

    def test_influence_matches_dense_formula_and_single_token_derivatives(self):
        q, k, v, qp, kp = _case(value_dim=3)
        generator = torch.Generator().manual_seed(19)
        dk = torch.randn(4, 2, 5, generator=generator) * 0.1
        dv = torch.randn(4, 2, 3, generator=generator) * 0.1
        weights, output = _dense(q, k, v, qp, kp)
        expanded_dk = dk.double().repeat_interleave(2, dim=1)
        expanded_dv = dv.double().repeat_interleave(2, dim=1)
        target_v = v[2:6].double().repeat_interleave(2, dim=1)
        delta_logits = torch.einsum("qhd,thd->qht", q.double(), expanded_dk)
        delta_logits /= math.sqrt(q.shape[-1])
        changes = weights[:, :, 2:6, None] * (
            expanded_dv.permute(1, 0, 2)[None]
            + delta_logits[..., None]
            * (target_v.permute(1, 0, 2)[None] - output[:, :, None, :])
        )
        expected = changes.square().sum(-1).sum(1).mean(0)
        for query_block, key_block in ((1, 2), (3, 4), (16, 256)):
            actual = scoring.compute_attention_influence(
                q, k, v, dk, dv,
                target_start=2, q_positions=qp, key_positions=kp,
                query_block_size=query_block, key_block_size=key_block,
            )
            torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=1e-7)

        epsilon = 1e-5
        derivatives = []
        for token in range(len(dk)):
            repaired_k, repaired_v = k.double().clone(), v.double().clone()
            repaired_k[2 + token] += epsilon * dk[token].double()
            repaired_v[2 + token] += epsilon * dv[token].double()
            _, repaired_output = _dense(q, repaired_k, repaired_v, qp, kp)
            derivative = (repaired_output - output) / epsilon
            derivatives.append(derivative.square().sum((1, 2)).mean())
        torch.testing.assert_close(
            actual.double(), torch.stack(derivatives), rtol=2e-4, atol=1e-7
        )

    def test_rank_one_residual_noise_is_zero_without_erasing_small_real_drift(self):
        q, k, v, qp, kp = _case(value_dim=3)
        generator = torch.Generator().manual_seed(37)
        dk = torch.randn(4, 2, 5, generator=generator)
        dv = torch.randn(4, 2, 3, generator=generator)
        kwargs = dict(target_start=2, q_positions=qp, key_positions=kp)
        coefficients = torch.tensor([-2.0, 0.37, 1.2, 5.0])[:, None, None]
        residual = scoring.compute_attention_influence(
            q, k, v, coefficients * dk[:1], coefficients * dv[:1],
            **kwargs, residual_weight=1.0,
        )
        self.assertEqual(torch.count_nonzero(residual).item(), 0)
        torch.testing.assert_close(scoring.percentile_rank(residual), torch.zeros(4))
        relevance = torch.tensor([3.0, 1.0, 7.0, 2.0])
        torch.testing.assert_close(
            scoring.fuse_scores(relevance, residual),
            0.5 * scoring.percentile_rank(relevance),
        )
        real = scoring.compute_attention_influence(q, k, v, dk, dv, **kwargs, residual_weight=1.0)
        small = scoring.compute_attention_influence(
            q, k, v, dk * 1e-12, dv * 1e-12, **kwargs, residual_weight=1.0
        )
        self.assertGreater(torch.count_nonzero(small).item(), 0)
        torch.testing.assert_close(small / 1e-24, real, rtol=2e-4, atol=1e-7)

    def test_rank_ties_constants_and_fusion_endpoints(self):
        relevance = torch.tensor([3.0, 1.0, 2.0, 2.0])
        influence = relevance.flip(0)
        ranks = torch.tensor([1.0, 0.0, 0.5, 0.5])
        torch.testing.assert_close(scoring.percentile_rank(relevance), ranks)
        torch.testing.assert_close(scoring.percentile_rank(torch.ones(4)), torch.zeros(4))
        torch.testing.assert_close(scoring.fuse_scores(relevance, influence, 0.0), ranks)
        torch.testing.assert_close(
            scoring.fuse_scores(relevance, influence, 1.0),
            scoring.percentile_rank(influence),
        )

    def test_constant_values_do_not_turn_key_only_drift_into_rank_noise(self):
        generator = torch.Generator().manual_seed(1)
        q = torch.randn(4, 4, 32, generator=generator)
        k = torch.randn(17, 2, 32, generator=generator)
        common_v = torch.randn(1, 2, 32, generator=generator)
        v = common_v.expand(17, -1, -1).clone()
        dk = torch.randn(9, 2, 32, generator=generator)
        dv = torch.zeros_like(dk)
        qp, kp = torch.arange(13, 17), torch.arange(17)
        kwargs = dict(
            target_start=2, q_positions=qp, key_positions=kp,
            key_block_size=5, query_block_size=2,
        )
        actual = scoring.compute_attention_influence(q, k, v, dk, dv, **kwargs)
        # Identical values make the attention output independent of every key.
        self.assertEqual(torch.count_nonzero(actual).item(), 0)
        torch.testing.assert_close(scoring.percentile_rank(actual), torch.zeros(9))

        # An O(eps) threshold on squared norms would discard these real signals.
        noise = torch.randn(v.shape, generator=generator)
        for magnitude in (1e-4, 1e-5):
            with self.subTest(magnitude=magnitude):
                varied_v = v + noise * magnitude
                weights, output = _dense(q, k, varied_v, qp, kp)
                delta_logits = torch.einsum(
                    "qhd,thd->qht", q.double(), dk.double().repeat_interleave(2, dim=1)
                ) / math.sqrt(q.shape[-1])
                difference = (
                    varied_v[2:11].double().repeat_interleave(2, dim=1)
                    .permute(1, 0, 2)[None]
                    - output[:, :, None, :]
                )
                changes = weights[:, :, 2:11, None] * delta_logits[..., None] * difference
                expected = changes.square().sum(-1).sum(1).mean(0)
                actual = scoring.compute_attention_influence(
                    q, k, varied_v, dk, dv, **kwargs
                )
                self.assertGreater(torch.count_nonzero(actual).item(), 0)
                torch.testing.assert_close(actual.double(), expected, rtol=0.01, atol=1e-13)

    def test_invalid_shapes_dtypes_ranges_and_nonfinite_scores_are_rejected(self):
        q, k, v, qp, kp = _case()
        invalid_calls = (
            lambda: attention.causal_attention(q, k, v, qp, kp, query_block_size=0),
            lambda: attention.causal_attention(q, k, v.double(), qp, kp),
            lambda: attention.causal_attention(q, k, v, qp.float(), kp),
            lambda: attention.causal_attention(q[:, :3], k, v, qp, kp),
            lambda: scoring.mean_attention_scores(
                q, k, v, q_positions=qp, key_positions=kp, key_block_size=-1
            ),
            lambda: scoring.compute_attention_influence(
                q, k, v, k[:2], v[:2], target_start=8,
                q_positions=qp, key_positions=kp,
            ),
            lambda: scoring.percentile_rank(torch.tensor([0.0, torch.nan])),
            lambda: scoring.fuse_scores(torch.zeros(2), torch.zeros(3)),
            lambda: scoring.fuse_scores(torch.zeros(2), torch.zeros(2), -0.1),
        )
        for call in invalid_calls:
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
