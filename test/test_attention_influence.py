"""CPU checks for query-conditioned selective KV repair scores."""

import importlib.util
import math
import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest import mock

import torch


# Keep the mathematical checks independent of SGLang and its GPU dependencies.
MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "srt" / "utils" / "attention_influence.py"
)
SPEC = importlib.util.spec_from_file_location("attention_influence", MODULE_PATH)
attention_influence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(attention_influence)
compute_attention_influence = attention_influence.compute_attention_influence
percentile_rank = attention_influence.percentile_rank
fuse_scores = attention_influence.fuse_scores


def dense_attention(q, k, v, q_positions, key_positions):
    """Independent, float64 full-attention reference with grouped query heads."""
    q, k, v = q.double(), k.double(), v.double()
    head_repeat = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(head_repeat, dim=1)
    v = v.repeat_interleave(head_repeat, dim=1)
    logits = torch.einsum("qhd,nhd->qhn", q, k) / math.sqrt(q.shape[-1])
    visible = key_positions[None, :] <= q_positions[:, None]
    logits = logits.masked_fill(~visible[:, None, :], -torch.inf)
    weights = logits.softmax(dim=-1)
    return weights, torch.einsum("qhn,nhv->qhv", weights, v)


def dense_influence(q, k, v, delta_k, delta_v, **kwargs):
    """Evaluate the first-order expression without streaming or PCA."""
    weights, output = dense_attention(
        q, k, v, kwargs["q_positions"], kwargs["key_positions"]
    )
    start = kwargs["target_start"]
    stop = start + delta_k.shape[0]
    head_repeat = q.shape[1] // k.shape[1]
    target_v = v[start:stop].double().repeat_interleave(head_repeat, dim=1)
    delta_k = delta_k.double().repeat_interleave(head_repeat, dim=1)
    delta_v = delta_v.double().repeat_interleave(head_repeat, dim=1)
    logit_change = torch.einsum("qhd,thd->qht", q.double(), delta_k)
    logit_change /= math.sqrt(q.shape[-1])
    change = weights[:, :, start:stop, None] * (
        delta_v.permute(1, 0, 2)[None]
        + logit_change[..., None]
        * (target_v.permute(1, 0, 2)[None] - output[:, :, None, :])
    )
    return change.square().sum(dim=-1).sum(dim=1).mean(dim=0)


class AttentionInfluenceTest(unittest.TestCase):
    def make_case(self, query_heads=4, kv_heads=2):
        generator = torch.Generator().manual_seed(42)

        def random_tensor(*shape):
            return torch.randn(*shape, generator=generator)

        args = (
            random_tensor(3, query_heads, 4),
            random_tensor(11, kv_heads, 4),
            random_tensor(11, kv_heads, 3),
            random_tensor(6, kv_heads, 4) * 0.1,
            random_tensor(6, kv_heads, 3) * 0.1,
        )
        kwargs = {
            "target_start": 2,
            "q_positions": torch.tensor([8, 13, 25]),
            "key_positions": torch.tensor([0, 2, 3, 7, 8, 11, 13, 14, 19, 22, 25]),
        }
        return args, kwargs

    def test_chunked_scores_match_dense_reference(self):
        # Includes unequal value/key dimensions and causal masking inside blocks.
        for query_heads, kv_heads in ((2, 2), (4, 2), (4, 1)):
            args, kwargs = self.make_case(query_heads, kv_heads)
            expected = dense_influence(*args, **kwargs)
            for key_block_size, query_block_size in ((1, 1), (4, 2), (32, 16)):
                with self.subTest(
                    query_heads=query_heads,
                    kv_heads=kv_heads,
                    key_block_size=key_block_size,
                    query_block_size=query_block_size,
                ):
                    actual = compute_attention_influence(
                        *args,
                        **kwargs,
                        key_block_size=key_block_size,
                        query_block_size=query_block_size,
                    )
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertEqual(tuple(actual.shape), (6,))
                    torch.testing.assert_close(
                        actual.double(), expected, rtol=2e-5, atol=1e-7
                    )

    def test_grouped_heads_match_explicit_kv_expansion(self):
        args, kwargs = self.make_case()
        q, k, v, delta_k, delta_v = args
        expanded = tuple(
            tensor.repeat_interleave(2, dim=1)
            for tensor in (k, v, delta_k, delta_v)
        )
        grouped_scores = compute_attention_influence(*args, **kwargs)
        expanded_scores = compute_attention_influence(q, *expanded, **kwargs)
        torch.testing.assert_close(grouped_scores, expanded_scores)

    def test_denominator_includes_prefix_and_query_keys(self):
        # Four visible keys: prefix, two targets, and the query's own key.
        # The remaining two keys are in the future and must have zero weight.
        actual = compute_attention_influence(
            torch.zeros(1, 1, 2),
            torch.zeros(6, 1, 2),
            torch.zeros(6, 1, 1),
            torch.zeros(2, 1, 2),
            torch.ones(2, 1, 1),
            target_start=1,
            q_positions=torch.tensor([5]),
            key_positions=torch.tensor([-5, 0, 4, 5, 10, 20]),
            key_block_size=2,
        )
        torch.testing.assert_close(actual, torch.full((2,), 1.0 / 16))

    def test_future_targets_have_zero_influence(self):
        actual = compute_attention_influence(
            torch.ones(1, 1, 2),
            torch.ones(4, 1, 2),
            torch.ones(4, 1, 3),
            torch.ones(2, 1, 2),
            torch.ones(2, 1, 3),
            target_start=2,
            q_positions=torch.tensor([5]),
            key_positions=torch.tensor([0, 5, 6, 7]),
            key_block_size=1,
        )
        torch.testing.assert_close(actual, torch.zeros(2))

    def test_small_single_token_repairs_match_finite_differences(self):
        args, kwargs = self.make_case()
        q, k, v, delta_k, delta_v = (tensor.double() for tensor in args)
        _, baseline = dense_attention(
            q, k, v, kwargs["q_positions"], kwargs["key_positions"]
        )
        epsilon = 1e-4
        expected = []
        for offset in range(delta_k.shape[0]):
            token = kwargs["target_start"] + offset
            repaired_k, repaired_v = k.clone(), v.clone()
            repaired_k[token] += epsilon * delta_k[offset]
            repaired_v[token] += epsilon * delta_v[offset]
            _, repaired_output = dense_attention(
                q,
                repaired_k,
                repaired_v,
                kwargs["q_positions"],
                kwargs["key_positions"],
            )
            derivative = (repaired_output - baseline) / epsilon
            expected.append(derivative.square().sum(dim=(1, 2)).mean())
        actual = compute_attention_influence(*args, **kwargs)
        torch.testing.assert_close(
            actual.double(), torch.stack(expected), rtol=2e-4, atol=1e-7
        )

    def test_zero_drift_is_zero_with_and_without_residualization(self):
        args, kwargs = self.make_case()
        q, k, v, delta_k, delta_v = args
        for residual_weight in (0.0, 1.0):
            with self.subTest(residual_weight=residual_weight):
                actual = compute_attention_influence(
                    q,
                    k,
                    v,
                    torch.zeros_like(delta_k),
                    torch.zeros_like(delta_v),
                    **kwargs,
                    residual_weight=residual_weight,
                )
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, torch.zeros_like(actual))

    def test_common_value_drift_is_removed_only_when_requested(self):
        args, kwargs = self.make_case()
        q, k, v, delta_k, delta_v = args
        delta_k = torch.zeros_like(delta_k)
        common = torch.tensor([[1.0, 2.0, -0.5], [-0.25, 1.5, 0.75]])
        delta_v = common.unsqueeze(0).expand_as(delta_v).clone()
        raw = compute_attention_influence(q, k, v, delta_k, delta_v, **kwargs)
        residual = compute_attention_influence(
            q, k, v, delta_k, delta_v, **kwargs, residual_weight=1.0
        )
        # Repairing one token's common value shift can affect output, even though
        # the optional shared-direction projection deliberately discards it.
        self.assertGreater(raw.min().item(), 0.0)
        self.assertLess(residual.abs().max().item(), 1e-9)

    def test_rank_one_roundoff_does_not_create_spurious_fusion_ranks(self):
        args, kwargs = self.make_case()
        q, k, v, delta_k, delta_v = args
        coefficients = torch.tensor([-3.0, -1.0, 0.25, 2.0, 4.0, 7.0])
        delta_k = coefficients[:, None, None] * delta_k[:1]
        delta_v = coefficients[:, None, None] * delta_v[:1]
        residual = compute_attention_influence(
            q, k, v, delta_k, delta_v, **kwargs, residual_weight=1.0
        )
        self.assertEqual(torch.count_nonzero(residual).item(), 0)
        torch.testing.assert_close(percentile_rank(residual), torch.zeros(6))
        relevance = torch.tensor([1.0, 4.0, 2.0, 6.0, 5.0, 3.0])
        torch.testing.assert_close(
            fuse_scores(relevance, residual), 0.5 * percentile_rank(relevance)
        )

    def test_small_real_residuals_are_not_removed_by_absolute_thresholds(self):
        args, kwargs = self.make_case()
        q, k, v, delta_k, delta_v = args
        residual = compute_attention_influence(
            *args, **kwargs, residual_weight=1.0
        )
        scale = 1e-8
        scaled = compute_attention_influence(
            q,
            k,
            v,
            delta_k * scale,
            delta_v * scale,
            **kwargs,
            residual_weight=1.0,
        )
        self.assertTrue(torch.all(residual > 0))
        torch.testing.assert_close(scaled / scale**2, residual, rtol=2e-4, atol=0)

    def test_invalid_shapes_and_ranges_are_rejected(self):
        args, kwargs = self.make_case()
        invalid_cases = []
        for index, tensor in (
            (0, args[0][..., 0]),
            (0, torch.zeros(3, 3, 4)),
            (1, args[1][..., :3]),
            (2, args[2][:-1]),
            (3, args[3][:-1]),
            (4, args[4][..., :2]),
        ):
            changed = list(args)
            changed[index] = tensor
            invalid_cases.append((changed, kwargs))
        for key, value in (
            ("target_start", -1),
            ("target_start", 6),
            ("q_positions", torch.tensor([1, 2])),
            ("key_positions", torch.tensor([1, 2])),
            ("key_block_size", 0),
            ("query_block_size", 0),
            ("residual_weight", -0.1),
            ("residual_weight", 1.1),
        ):
            invalid_cases.append((args, {**kwargs, key: value}))
        for index, (invalid_args, invalid_kwargs) in enumerate(invalid_cases):
            with self.subTest(case=index):
                with self.assertRaises((ValueError, TypeError)):
                    compute_attention_influence(*invalid_args, **invalid_kwargs)


class RankFusionTest(unittest.TestCase):
    def test_percentile_rank_uses_average_ranks_for_ties(self):
        actual = percentile_rank(torch.tensor([5.0, 9.0, 5.0, 7.0]))
        torch.testing.assert_close(
            actual, torch.tensor([1.0 / 6, 1.0, 1.0 / 6, 2.0 / 3])
        )

    def test_constant_and_singleton_ranks_are_zero(self):
        for values in (torch.ones(4), torch.tensor([7.0])):
            with self.subTest(size=values.numel()):
                torch.testing.assert_close(
                    percentile_rank(values), torch.zeros_like(values)
                )

    def test_fusion_endpoints_and_midpoint(self):
        relevance = torch.tensor([3.0, 5.0, 3.0, 4.0])
        influence = torch.tensor([20.0, 10.0, 30.0, 40.0])
        relevance_rank = percentile_rank(relevance)
        influence_rank = percentile_rank(influence)
        for weight in (0.0, 0.5, 1.0):
            with self.subTest(weight=weight):
                torch.testing.assert_close(
                    fuse_scores(relevance, influence, influence_weight=weight),
                    (1 - weight) * relevance_rank + weight * influence_rank,
                )

    def test_fusion_rejects_mismatched_scores_and_invalid_weights(self):
        for influence, weight in (
            (torch.ones(2), 0.5),
            (torch.ones(3), -0.1),
            (torch.ones(3), 1.1),
        ):
            with self.subTest(influence_shape=tuple(influence.shape), weight=weight):
                with self.assertRaises((ValueError, TypeError)):
                    fuse_scores(torch.ones(3), influence, influence_weight=weight)


def dense_runtime_relevance(q, k, *, target_start, target_len, q_start):
    """CPU replacement for the Triton kernel, preserving its score definition."""
    per_layer = []
    for layer in range(q.shape[0]):
        weights, _ = dense_attention(
            q[layer],
            k[layer],
            torch.zeros_like(k[layer]),
            torch.arange(q_start, q_start + q.shape[1]),
            torch.arange(k.shape[1]),
        )
        per_layer.append(
            weights[:, :, target_start : target_start + target_len].amax(dim=(0, 1))
        )
    return torch.stack(per_layer).mean(dim=0).float()


@contextmanager
def load_runtime_modules(include_blender=False):
    """Load real selector code with temporary dependency stubs, then restore imports."""
    replacements = {}
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.utils",
        "sglang.srt.layers",
        "sglang.srt.model_executor",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        replacements[name] = package
    replacements["sglang.srt.utils.attention_influence"] = attention_influence
    scorer = types.ModuleType("sglang.srt.utils.triton_attention_score")
    scorer.compute_att_full_softmax_importance = mock.Mock(
        wraps=dense_runtime_relevance
    )
    replacements[scorer.__name__] = scorer

    with mock.patch.dict(sys.modules, replacements):
        def load(name, relative_path):
            spec = importlib.util.spec_from_file_location(
                name, MODULE_PATH.parents[2] / relative_path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        info = load(
            "sglang.srt.utils.cache_blender_info", "srt/utils/cache_blender_info.py"
        )
        selector = load("sglang.srt.utils.indice_select", "srt/utils/indice_select.py")
        runtime = types.SimpleNamespace(info=info, selector=selector, scorer=scorer)
        if include_blender:
            dependencies = {
                "sglang.srt.layers.rotary_embedding": {"RotaryEmbedding": object},
                "sglang.srt.model_executor.forward_batch_info": {
                    "ForwardBatch": object
                },
                "sglang.srt.utils.digest_index_manager": {"DigestIndexManager": object},
            }
            ssd = types.SimpleNamespace(
                is_online=mock.Mock(return_value=True),
                wait_task_b=mock.Mock(
                    side_effect=AssertionError("Unexpected Task B wait")
                ),
                wait_layer_ready=mock.Mock(
                    side_effect=AssertionError("Unexpected probe cache wait")
                ),
            )
            dependencies["sglang.srt.utils.kv_ssd_manager"] = {
                "KVSSDManager": ssd,
                "hack_pool_lock": nullcontext(),
                "context_pool_lock": nullcontext(),
            }
            for name, attributes in dependencies.items():
                dependency = types.ModuleType(name)
                dependency.__dict__.update(attributes)
                sys.modules[name] = dependency
            runtime.blender = load(
                "sglang.srt.utils.cache_blender", "srt/utils/cache_blender.py"
            )
            runtime.ssd = ssd
        yield runtime


class RuntimeSelectorTest(unittest.TestCase):
    def make_info(self, runtime):
        info = runtime.info.BatchBlendInfo()
        info.att_params = runtime.info.AttParams(
            num_heads=4, num_kv_heads=2, head_dim=2, num_layers=4
        )
        info.select_mode = runtime.info.SelectMode.INFLUENCE
        info.start = 1
        info.ratio = 0.5
        info.attn_start = 0
        info.attn_end = 2
        info.chunk_loc_list = torch.tensor([0, 1, 5, 8])
        info.req_len_list = torch.tensor([3])
        return info

    def test_runtime_probe_matches_dense_and_uses_entire_current_suffix(self):
        with load_runtime_modules() as runtime:
            info = self.make_info(runtime)
            region = runtime.selector._ReqRegion(2, 8, 11, 4, 4)
            generator = torch.Generator().manual_seed(73)
            probe = (
                torch.randn(11, 8, generator=generator),
                torch.randn(11, 4, generator=generator),
                torch.randn(11, 4, generator=generator),
                torch.randn(8, 4, generator=generator),
                torch.randn(8, 4, generator=generator),
            )
            positions = torch.tensor([90, 91, 0, 1, 2, 3, 4, 5, 6, 7, 8])
            # QCOMPUTE token offsets must never index the current dense suffix.
            runtime.info.HackBlendKVPool.q_lens = [1]
            runtime.info.HackBlendKVPool.q_offsets = [100]
            q, k, v, cached_k, cached_v = probe
            expected = dense_influence(
                q[8:11].reshape(3, 4, 2),
                torch.cat((cached_k[2:8], k[8:11])).reshape(9, 2, 2),
                torch.cat((cached_v[2:8], v[8:11])).reshape(9, 2, 2),
                (k[4:8] - cached_k[4:8]).reshape(4, 2, 2),
                (v[4:8] - cached_v[4:8]).reshape(4, 2, 2),
                target_start=2,
                q_positions=positions[8:11],
                key_positions=positions[2:11],
            )
            actual = runtime.selector.IndiceSelector._compute_influence(
                info, region, positions, probe
            )
            torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=1e-6)

    def test_in_place_rotary_preserves_probe_and_cache_tensors(self):
        with load_runtime_modules() as runtime:
            info = self.make_info(runtime)
            generator = torch.Generator().manual_seed(19)
            probe = tuple(
                torch.randn(8, width, generator=generator)
                for width in (8, 4, 4, 4, 4)
            )
            originals = tuple(tensor.clone() for tensor in probe)
            positions = torch.arange(8)
            region = runtime.selector._ReqRegion(0, 5, 8, 1, 4)

            def inplace_rotary(pos, q, k):
                self.assertNotEqual(q.data_ptr(), k.data_ptr())
                q.add_(pos[:, None] * 0.1)
                k.mul_(1.0 + pos[:, None] * 0.01)
                return q, k

            info.rotary_emb = inplace_rotary
            q, k, v, cached_k, cached_v = originals
            reference_k = torch.cat((cached_k[:5], k[5:]))
            rotated_k = reference_k * (1.0 + positions[:, None] * 0.01)
            rotated_new_k = k[1:5] * (1.0 + positions[1:5, None] * 0.01)
            expected = dense_influence(
                (q[5:] + positions[5:, None] * 0.1).reshape(3, 4, 2),
                rotated_k.reshape(8, 2, 2),
                torch.cat((cached_v[:5], v[5:])).reshape(8, 2, 2),
                (rotated_new_k - rotated_k[1:5]).reshape(4, 2, 2),
                (v[1:5] - cached_v[1:5]).reshape(4, 2, 2),
                target_start=1,
                q_positions=positions[5:],
                key_positions=positions,
            )
            actual = runtime.selector.IndiceSelector._compute_influence(
                info, region, positions, probe
            )
            torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=1e-6)
            for tensor, original in zip(probe, originals):
                torch.testing.assert_close(tensor, original, rtol=0, atol=0)

    def test_runtime_denominator_includes_system_and_current_query(self):
        with load_runtime_modules() as runtime:
            info = self.make_info(runtime)
            info.att_params = runtime.info.AttParams(1, 1, 1, 4)
            current_v = torch.tensor([[0.0], [1.0], [1.0], [1.0], [0.0], [0.0]])
            zeros = torch.zeros(6, 1)
            actual = runtime.selector.IndiceSelector._compute_influence(
                info,
                runtime.selector._ReqRegion(0, 4, 6, 1, 3),
                torch.arange(6),
                (zeros, zeros, current_v, zeros, zeros),
            )
            expected = torch.full((3,), (1.0 / 5**2 + 1.0 / 6**2) / 2)
            torch.testing.assert_close(actual, expected)

    def test_budget_boundaries_and_query_only_requests(self):
        with load_runtime_modules() as runtime:
            for length, ratio, expected in (
                (0, 1, 0), (5, 0, 0), (5, 0.01, 1), (5, 1, 5)
            ):
                self.assertEqual(
                    runtime.selector._compute_budget(length, ratio), expected
                )
            info = self.make_info(runtime)
            info.ratio = 0
            info.chunk_loc_list = torch.tensor([0, 2, 6, 9, 10, 12])
            info.req_len_list = torch.tensor([3, 2])
            indices, lengths = runtime.selector.IndiceSelector.select(info)
            torch.testing.assert_close(indices, torch.tensor([6, 7, 8, 10, 11]))
            torch.testing.assert_close(lengths, torch.tensor([3, 2]))

    def test_full_selector_preserves_budget_suffix_and_attention_baseline(self):
        with load_runtime_modules() as runtime:
            info = self.make_info(runtime)
            generator = torch.Generator().manual_seed(5)
            old_k = [torch.randn(8, 4, generator=generator) for _ in range(2)]
            old_q = [torch.randn(2, 8, generator=generator) for _ in range(2)]
            query_k = [torch.randn(3, 4, generator=generator) for _ in range(2)]
            pool = runtime.info.HackBlendKVPool
            pool.q_lens, pool.q_offsets, pool.query_k_lens = [2], [1], [3]
            pool.query_k_buffer = query_k
            probe = tuple(
                torch.randn(8, width, generator=generator)
                for width in (8, 4, 4, 4, 4)
            )
            q, k, v, cached_k, cached_v = probe
            positions = torch.arange(8)
            q_stacked = torch.stack(old_q).reshape(2, 2, 4, 2)
            k_stacked = torch.stack(
                [
                    torch.cat((layer_k[:5], suffix))
                    for layer_k, suffix in zip(old_k, query_k)
                ]
            ).reshape(2, 8, 2, 2)
            relevance = dense_runtime_relevance(
                q_stacked, k_stacked, target_start=1, target_len=4, q_start=6
            )
            influence = dense_influence(
                q[5:].reshape(3, 4, 2),
                torch.cat((cached_k[:5], k[5:])).reshape(8, 2, 2),
                torch.cat((cached_v[:5], v[5:])).reshape(8, 2, 2),
                (k[1:5] - cached_k[1:5]).reshape(4, 2, 2),
                (v[1:5] - cached_v[1:5]).reshape(4, 2, 2),
                target_start=1,
                q_positions=positions[5:],
                key_positions=positions,
            ).float()
            probe_kwargs = dict(
                current_q=q,
                current_k=k,
                current_v=v,
                cached_k=cached_k,
                cached_v=cached_v,
            )
            # Only bypass the CUDA dispatch guard. Selection and FO stay real;
            # the unavailable Triton kernel uses an independent dense formula.
            with mock.patch.object(
                torch.Tensor,
                "is_cuda",
                new_callable=mock.PropertyMock,
                return_value=True,
            ):
                for mode, expected_scores in (
                    (runtime.info.SelectMode.ATTN, relevance),
                    (
                        runtime.info.SelectMode.INFLUENCE,
                        fuse_scores(relevance, influence),
                    ),
                ):
                    info.select_mode = mode
                    for ratio, budget in ((0.01, 1), (0.5, 2), (1.0, 4)):
                        with self.subTest(mode=mode, ratio=ratio):
                            info.ratio = ratio
                            indices, lengths = runtime.selector.IndiceSelector.select(
                                info, old_k, old_q, positions, **probe_kwargs
                            )
                            selected = (
                                torch.topk(expected_scores, budget).indices.sort().values
                                + 1
                            )
                            expected = torch.cat((selected, torch.arange(5, 8)))
                            torch.testing.assert_close(indices, expected)
                            torch.testing.assert_close(
                                lengths, torch.tensor([budget + 3])
                            )
            self.assertEqual(
                runtime.scorer.compute_att_full_softmax_importance.call_count, 6
            )

    def test_zero_ratio_blender_does_not_wait_for_or_read_probe_caches(self):
        with load_runtime_modules(include_blender=True) as runtime:
            info = self.make_info(runtime)
            info.ratio = 0
            info.blend_style = runtime.info.BlendStyle.DO_BLEND
            q, k, v = torch.randn(8, 8), torch.randn(8, 4), torch.randn(8, 4)
            with mock.patch.object(
                runtime.info.HackBlendKVPool,
                "get_kv",
                side_effect=AssertionError("Unexpected probe cache read"),
            ), mock.patch.object(
                runtime.info.HackBlendKVPool,
                "get_all_kv",
                side_effect=AssertionError("Unexpected attention cache read"),
            ):
                actual_q, actual_k, actual_v = runtime.blender.CacheBlender.blend(
                    1, q, k, v, torch.arange(8),
                    types.SimpleNamespace(blend_info=info), None
                )
            runtime.ssd.wait_task_b.assert_not_called()
            runtime.ssd.wait_layer_ready.assert_not_called()
            torch.testing.assert_close(actual_q, q[5:])
            torch.testing.assert_close(actual_k, k)
            torch.testing.assert_close(actual_v, v)
            torch.testing.assert_close(info.blend_top_indices, torch.arange(5, 8))
            self.assertEqual(info.influence_metrics["selected_doc_tokens"], 0)

    def test_blender_uses_same_layer_probe_then_scatters_selected_kv_safely(self):
        with load_runtime_modules(include_blender=True) as runtime:
            info = self.make_info(runtime)
            info.blend_style = runtime.info.BlendStyle.DO_BLEND
            info.influence_weight = 1.0
            info.keep_layers_set = {2}
            runtime.ssd.wait_task_b = mock.Mock()
            runtime.ssd.wait_layer_ready = mock.Mock()
            generator = torch.Generator().manual_seed(37)

            def random_tensor(rows, width):
                return torch.randn(rows, width, generator=generator)

            pool = runtime.info.HackBlendKVPool
            pool.k_buffer = [random_tensor(8, 4) for _ in range(3)]
            pool.v_buffer = [random_tensor(8, 4) for _ in range(3)]
            pool.q_buffer = [random_tensor(2, 8) for _ in range(2)]
            pool.query_k_buffer = [random_tensor(3, 4) for _ in range(2)]
            pool.q_lens, pool.q_offsets, pool.query_k_lens = [2], [1], [3]
            q, k, v = random_tensor(8, 8), random_tensor(8, 4), random_tensor(8, 4)
            positions = torch.arange(8)
            cached_k, cached_v = pool.get_kv(1)
            expected_scores = dense_influence(
                q[5:].reshape(3, 4, 2),
                torch.cat((cached_k[:5], k[5:])).reshape(8, 2, 2),
                torch.cat((cached_v[:5], v[5:])).reshape(8, 2, 2),
                (k[1:5] - cached_k[1:5]).reshape(4, 2, 2),
                (v[1:5] - cached_v[1:5]).reshape(4, 2, 2),
                target_start=1,
                q_positions=positions[5:],
                key_positions=positions,
            )
            expected_indices = torch.cat(
                (
                    torch.topk(expected_scores, 2).indices.sort().values + 1,
                    positions[5:],
                )
            )
            batch = types.SimpleNamespace(blend_info=info)
            with mock.patch.object(
                torch.Tensor,
                "is_cuda",
                new_callable=mock.PropertyMock,
                return_value=True,
            ), mock.patch.object(
                runtime.selector.IndiceSelector,
                "_compute_influence",
                wraps=runtime.selector.IndiceSelector._compute_influence,
            ) as influence_spy:
                out_q, out_k, out_v = runtime.blender.CacheBlender.blend(
                    1, q, k, v, positions, batch, None
                )
            runtime.ssd.wait_task_b.assert_called_once_with()
            runtime.ssd.wait_layer_ready.assert_called_once_with(1)
            influence_spy.assert_called_once()
            actual_probe = influence_spy.call_args.args[3]
            self.assertIs(actual_probe[3], cached_k)
            self.assertIs(actual_probe[4], cached_v)
            torch.testing.assert_close(info.blend_top_indices, expected_indices)
            torch.testing.assert_close(out_q, q[expected_indices])
            torch.testing.assert_close(out_k, k)
            torch.testing.assert_close(out_v, v)

            # A later layer replaces only selected KV rows. Its retained cache
            # must remain raw and unchanged for subsequent ratio evaluations.
            next_q = random_tensor(5, 8)
            next_k, next_v = random_tensor(5, 4), random_tensor(5, 4)
            saved_k, saved_v = pool.k_buffer[2].clone(), pool.v_buffer[2].clone()
            expected_k, expected_v = saved_k.clone(), saved_v.clone()
            expected_k[expected_indices] = next_k
            expected_v[expected_indices] = next_v
            later_q, later_k, later_v = runtime.blender.CacheBlender.blend(
                2, next_q, next_k, next_v, positions, batch, None
            )
            self.assertEqual(
                runtime.ssd.wait_layer_ready.call_args_list,
                [mock.call(1), mock.call(2)],
            )
            torch.testing.assert_close(later_q, next_q)
            torch.testing.assert_close(later_k, expected_k)
            torch.testing.assert_close(later_v, expected_v)
            torch.testing.assert_close(pool.k_buffer[2], saved_k, rtol=0, atol=0)
            torch.testing.assert_close(pool.v_buffer[2], saved_v, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
