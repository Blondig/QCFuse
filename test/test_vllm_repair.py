"""CPU integration checks using a small model with vLLM decoder interfaces.

These verify numerical/cache semantics; they do not replace a CUDA vLLM smoke run.
"""

import math
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from vllm_repair.runtime import RepairConfig, SparseRepairRuntime
from vllm_repair.worker import RepairWorkerExtension, _make_kv_writer
from vllm_repair.run import measure_request, tokenize_request


class TupleLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
        normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        normalized = normalized * self.weight
        return normalized if residual is None else (normalized, x)


class InplaceRoPE(nn.Module):
    """Deliberately mutates Q/K, like a native rotary kernel may do."""

    @staticmethod
    def rotate(x, positions):
        shape = x.shape
        x = x.reshape(len(positions), -1, 4)
        angle = positions[:, None, None].to(x.dtype) * x.new_tensor([0.17, 0.031])
        a, b = x[..., :2].clone(), x[..., 2:].clone()
        return torch.cat((a * angle.cos() - b * angle.sin(),
                          b * angle.cos() + a * angle.sin()), dim=-1).reshape(shape)

    def forward(self, positions, q, k):
        q.copy_(self.rotate(q, positions))
        k.copy_(self.rotate(k, positions))
        return q, k


class ToyAttention(nn.Module):
    def __init__(self, qk_norm):
        super().__init__()
        self.num_heads = 4
        self.num_kv_heads = 2
        self.head_dim = 4
        self.head_size = 4
        self.q_size = 16
        self.kv_size = 8
        self.qkv_proj = TupleLinear(16, 32, bias=True)
        self.o_proj = TupleLinear(16, 16, bias=False)
        self.rotary_emb = InplaceRoPE()
        if qk_norm:
            self.q_norm = RMSNorm(4)
            self.k_norm = RMSNorm(4)


class ToyLayer(nn.Module):
    def __init__(self, qk_norm):
        super().__init__()
        self.self_attn = ToyAttention(qk_norm)
        self.input_layernorm = RMSNorm(16)
        self.post_attention_layernorm = RMSNorm(16)
        self.mlp = nn.Sequential(nn.Linear(16, 24), nn.SiLU(), nn.Linear(24, 16))


class ToyModel(nn.Module):
    def __init__(self, qk_norm=True):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(451)
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(64, 16)
            self.model.layers = nn.ModuleList([ToyLayer(qk_norm) for _ in range(4)])
            self.model.norm = RMSNorm(16)
        self.config = SimpleNamespace(hidden_size=16, num_hidden_layers=4,
                                      model_type="qwen3" if qk_norm else "llama")
        self.model.start_layer = 0
        self.model.end_layer = 4


@torch.no_grad()
def full_reference(model, ids, positions):
    """Conventional dense decoder, separate from the sparse runtime and SDPA."""
    x = model.model.embed_tokens(ids)
    caches = []
    for layer in model.model.layers:
        attn = layer.self_attn
        normalized = layer.input_layernorm(x)
        qkv = nn.functional.linear(normalized, attn.qkv_proj.weight, attn.qkv_proj.bias)
        q, k, v = qkv.split([16, 8, 8], dim=-1)
        q, k, v = q.reshape(-1, 4, 4), k.reshape(-1, 2, 4), v.reshape(-1, 2, 4)
        if hasattr(attn, "q_norm"):
            q, k = attn.q_norm(q), attn.k_norm(k)
        q = InplaceRoPE.rotate(q, positions)
        k = InplaceRoPE.rotate(k, positions)
        caches.append((k.clone(), v.clone()))
        keys, values = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
        logits = torch.einsum("qhd,khd->hqk", q.double(), keys.double()) / 2
        allowed = positions[:, None] >= positions[None, :]
        weights = logits.masked_fill(~allowed, -math.inf).softmax(-1)
        attended = torch.einsum("hqk,khd->qhd", weights, values.double()).float()
        x = x + nn.functional.linear(attended.flatten(1), attn.o_proj.weight)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return model.model.norm(x), caches


class RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.prefix = [1, 2]
        self.docs = [[3, 4, 5, 6], [7, 8, 9, 10]]
        self.suffix = [11, 12, 13]
        self.ids = torch.tensor(self.prefix + sum(self.docs, []) + self.suffix)
        self.positions = torch.arange(len(self.ids))

    def execute(self, model, config, runtime=None):
        runtime = runtime or SparseRepairRuntime(model)
        runtime.prepare_chunks([self.prefix] + self.docs)
        runtime.arm(self.prefix, self.docs, self.suffix, config)
        written = {}

        def write_kv(layer_id, k, v):
            written[layer_id] = (k.clone(), v.clone())

        output = runtime.run_prefill(self.ids, self.positions, write_kv)
        self.assertIsNone(runtime.pending)
        self.assertEqual(set(written), set(range(4)))
        for k, v in written.values():
            self.assertEqual(k.shape, (len(self.ids), 2, 4))
            self.assertEqual(v.shape, (len(self.ids), 2, 4))
            self.assertTrue(torch.isfinite(k).all())
            self.assertTrue(torch.isfinite(v).all())
        return output, written, runtime

    def test_full_budget_matches_dense_model_and_every_layer_cache(self):
        for qk_norm in (False, True):
            for method in ("prophet", "prophet_fo", "prophet_fo_residual"):
                with self.subTest(qk_norm=qk_norm, method=method):
                    model = ToyModel(qk_norm)
                    expected, caches = full_reference(model, self.ids, self.positions)
                    actual, written, runtime = self.execute(
                        model, RepairConfig(method=method, ratio=1, probe_layer=2))
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                    for index, (k, v) in enumerate(caches):
                        torch.testing.assert_close(written[index][0], k, atol=2e-6, rtol=2e-5)
                        torch.testing.assert_close(written[index][1], v, atol=2e-6, rtol=2e-5)

    def test_sparse_repair_all_modes_write_full_caches_and_finite_suffix(self):
        for method in ("prophet", "prophet_fo", "prophet_fo_residual", "prophet_fo_mixed"):
            with self.subTest(method=method):
                actual, _, _ = self.execute(
                    ToyModel(), RepairConfig(method=method, ratio=.25, probe_layer=2))
                self.assertEqual(actual.shape, (len(self.ids), 16))
                self.assertTrue(torch.isfinite(actual).all())
                self.assertGreater(float(actual[-1].norm()), 0)

    def test_reused_cache_is_not_rotated_or_polluted_by_other_queries(self):
        model = ToyModel()
        config = RepairConfig(method="prophet_fo_mixed", ratio=.25, probe_layer=1)
        original, cache1, runtime = self.execute(model, config)
        self.docs = list(reversed(self.docs))
        self.ids = torch.tensor(self.prefix + sum(self.docs, []) + self.suffix)
        self.execute(model, config, runtime)
        self.docs = list(reversed(self.docs))
        self.ids = torch.tensor(self.prefix + sum(self.docs, []) + self.suffix)
        repeated, cache2, _ = self.execute(model, config, runtime)
        torch.testing.assert_close(repeated, original, atol=0, rtol=0)
        for index in cache1:
            torch.testing.assert_close(cache2[index][0], cache1[index][0], atol=0, rtol=0)
            torch.testing.assert_close(cache2[index][1], cache1[index][1], atol=0, rtol=0)

    def test_shallow_layers_match_dense_even_with_zero_document_budget(self):
        model = ToyModel()
        _, dense = full_reference(model, self.ids, self.positions)
        _, caches, _ = self.execute(model, RepairConfig(ratio=0, probe_layer=2))
        # The measurement layer (index p) projects all K/V before sparsification.
        for index in range(3):
            torch.testing.assert_close(caches[index][0], dense[index][0], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(caches[index][1], dense[index][1], atol=2e-6, rtol=2e-5)

    def test_bad_prompt_clears_pending_request(self):
        runtime = SparseRepairRuntime(ToyModel())
        runtime.prepare_chunks([self.prefix] + self.docs)
        runtime.arm(self.prefix, self.docs, self.suffix, RepairConfig())
        with self.assertRaises((ValueError, RuntimeError)):
            runtime.run_prefill(self.ids.flip(0), self.positions, lambda *args: None)
        self.assertIsNone(runtime.pending)

    def test_budget_and_actual_projection_mlp_work(self):
        for method, probe_layer, ratio in (("prophet", 0, .25),
                                           ("prophet_fo", 2, .25),
                                           ("prophet_fo", 2, 0)):
            with self.subTest(method=method, probe_layer=probe_layer, ratio=ratio):
                model = ToyModel()
                runtime = SparseRepairRuntime(model)
                runtime.prepare_chunks([self.prefix] + self.docs)
                projection_sizes = [[] for _ in range(4)]
                mlp_sizes = [[] for _ in range(4)]
                handles = []
                for i, layer in enumerate(model.model.layers):
                    handles.append(layer.self_attn.qkv_proj.register_forward_pre_hook(
                        lambda _m, args, i=i: projection_sizes[i].append(len(args[0]))))
                    handles.append(layer.mlp.register_forward_pre_hook(
                        lambda _m, args, i=i: mlp_sizes[i].append(len(args[0]))))
                config = RepairConfig(method=method, ratio=ratio, probe_layer=probe_layer)
                self.execute(model, config, runtime)
                for handle in handles:
                    handle.remove()
                selected_docs = int(8 * ratio)
                selected_total = len(self.prefix) + selected_docs + len(self.suffix)
                probe_sizes = [len(self.suffix)] if ratio else []
                for i in range(4):
                    full_projection = probe_layer > 0 and i <= probe_layer
                    full_mlp = i < probe_layer
                    self.assertEqual(projection_sizes[i], probe_sizes + [len(self.ids) if full_projection else selected_total])
                    self.assertEqual(mlp_sizes[i], probe_sizes + [len(self.ids) if full_mlp else selected_total])
                selected = runtime.metrics["selected_positions"]
                self.assertEqual(selected[:2], [0, 1])
                self.assertEqual(selected[-3:], [10, 11, 12])
                self.assertEqual(len(selected), selected_total)
                self.assertEqual(runtime.metrics["selected_doc_tokens"], selected_docs)
                self.assertAlmostEqual(runtime.metrics["doc_attention_mlp_fraction"],
                                       (probe_layer * 8 + (4 - probe_layer) * selected_docs) / 32)

    def test_selection_uses_both_prophet_and_fo_rank(self):
        model = ToyModel()
        for weight, expected_docs in ((0, [8, 9]), (1, [2, 3])):
            runtime = SparseRepairRuntime(model)
            with mock.patch.object(runtime, "_query_probe", return_value=torch.arange(8).float()), \
                 mock.patch.object(runtime, "_fo_scores", return_value=torch.arange(8).flip(0).float()):
                _, _, runtime = self.execute(model, RepairConfig(
                    ratio=.25, probe_layer=1, influence_weight=weight), runtime)
                self.assertEqual(runtime.metrics["selected_positions"][2:-3], expected_docs)

    def test_zero_full_and_no_document_requests_skip_selector(self):
        for ratio in (0, 1):
            runtime = SparseRepairRuntime(ToyModel())
            with mock.patch.object(runtime, "_query_probe", side_effect=AssertionError("probe should be skipped")), \
                 mock.patch.object(runtime, "_fo_scores", side_effect=AssertionError("FO should be skipped")):
                self.execute(runtime.model, RepairConfig(ratio=ratio), runtime)
        self.docs = []
        self.ids = torch.tensor(self.prefix + self.suffix)
        self.positions = torch.arange(len(self.ids))
        model = ToyModel()
        expected, _ = full_reference(model, self.ids, self.positions)
        output, _, runtime = self.execute(model, RepairConfig(ratio=.2))
        torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
        self.assertEqual(runtime.metrics["selected_doc_tokens"], 0)

    def test_cache_is_bounded_to_requested_chunks(self):
        runtime = SparseRepairRuntime(ToyModel())
        first = runtime.prepare_chunks([self.prefix] + self.docs)
        repeated = runtime.prepare_chunks([self.prefix] + self.docs + self.docs)
        self.assertEqual(first["new_chunks"], 3)
        self.assertEqual(repeated["new_chunks"], 0)
        self.assertEqual(repeated["reused_chunks"], 3)
        changed = runtime.prepare_chunks([self.prefix, [21, 22]])
        self.assertEqual(changed["cached_chunks"], 2)
        self.assertEqual(changed["reused_chunks"], 1)
        self.assertEqual(set(runtime.cache), {tuple(self.prefix), (21, 22)})

    def test_invalid_configuration_and_missing_cache_fail_early(self):
        for kwargs in ({"ratio": -.1}, {"ratio": float("nan")},
                       {"probe_layer": 0}, {"probe_layer": True},
                       {"influence_weight": 2}, {"method": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RepairConfig(**kwargs)
        runtime = SparseRepairRuntime(ToyModel())
        with self.assertRaisesRegex(ValueError, "prepare_chunks"):
            runtime.arm(self.prefix, self.docs, self.suffix, RepairConfig())
        with self.assertRaisesRegex(ValueError, "probe_layer"):
            runtime.arm(self.prefix, self.docs, self.suffix, RepairConfig(probe_layer=4))
        self.assertIsNone(runtime.pending)


class WorkerTest(unittest.TestCase):
    """Mock only vLLM allocation/kernel APIs; exercise the actual repair runtime."""

    def setUp(self):
        torch.set_num_threads(1)
        self.prefix, self.docs, self.suffix = [1, 2], [[3, 4, 5], [6, 7]], [8, 9]
        self.ids = torch.tensor(self.prefix + sum(self.docs, []) + self.suffix)
        self.positions = torch.arange(len(self.ids))
        model_type = type("Qwen3ForCausalLM", (ToyModel,), {
            "__module__": "vllm.model_executor.models.qwen3",
            "forward": lambda model, input_ids, positions, **kwargs: model.model.embed_tokens(input_ids),
        })
        self.model = model_type()
        impl_type = type("FlashAttentionImpl", (), {"__module__": "vllm.v1.attention.backends.flash_attn"})
        self.slots = torch.tensor([9, 1, 5, 2, 10, 3, 8, 11, 6])
        self.meta = SimpleNamespace(num_actual_tokens=9, query_start_loc=torch.tensor([0, 9]),
                                    seq_lens=torch.tensor([9]), slot_mapping=self.slots, use_cascade=False)
        meta_by_name = {}
        for i, layer in enumerate(self.model.model.layers):
            impl = impl_type()
            impl.kv_cache_dtype = "auto"
            impl.sliding_window = (-1, -1)
            layer.self_attn.attn = SimpleNamespace(
                impl=impl, layer_name=f"model.layers.{i}.self_attn.attn", num_kv_heads=2, head_size=4,
                attn_type="decoder", sliding_window=None,
                kv_cache=[torch.zeros(2, 3, 4, 2, 4)], _k_scale=torch.tensor(1.), _v_scale=torch.tensor(1.))
            meta_by_name[layer.self_attn.attn.layer_name] = self.meta
        self.context = SimpleNamespace(attn_metadata=meta_by_name, virtual_engine=0)
        self.kernel_calls = []

        def native_writer(k, v, kcache, vcache, slots, dtype, kscale, vscale):
            self.kernel_calls.append((k.clone(), v.clone()))
            kcache.reshape(-1, 2, 4)[slots] = k
            vcache.reshape(-1, 2, 4)[slots] = v

        fake_modules = {
            "vllm": SimpleNamespace(__version__="0.9.2", envs=SimpleNamespace(VLLM_USE_V1=True)),
            "vllm.forward_context": SimpleNamespace(get_forward_context=lambda: self.context),
            "vllm.attention.utils.fa_utils": SimpleNamespace(reshape_and_cache_flash=native_writer),
        }
        self.patcher = mock.patch.dict(sys.modules, fake_modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.worker = RepairWorkerExtension()
        self.worker.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_seqs=1, enable_chunked_prefill=False),
            cache_config=SimpleNamespace(enable_prefix_caching=False, cache_dtype="auto"),
            model_config=SimpleNamespace(enforce_eager=True, quantization=None, hf_config=SimpleNamespace(is_causal=True)),
            compilation_config=SimpleNamespace(level=0))
        request = SimpleNamespace(num_computed_tokens=0, prompt_token_ids=self.ids.tolist(),
                                  output_token_ids=[], sampling_params=SimpleNamespace(prompt_logprobs=None),
                                  mm_inputs=[], lora_request=None)
        self.worker.model_runner = SimpleNamespace(model=self.model,
            input_batch=SimpleNamespace(req_id_to_index={"request": 0}), requests={"request": request})

    def arm(self, ratio=.25):
        self.worker.repair_initialize()
        self.worker.repair_prepare_chunks([self.prefix] + self.docs)
        self.worker.repair_arm(self.prefix, self.docs, self.suffix,
                               {"ratio": ratio, "probe_layer": 1})

    def test_prefill_writes_real_slots_and_decode_uses_original_forward(self):
        self.arm(ratio=1)
        expected, caches = full_reference(self.model, self.ids, self.positions)
        output = self.model(self.ids, self.positions)
        torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
        self.assertEqual(len(self.kernel_calls), 4)
        unused = torch.tensor([0, 4, 7])
        for layer, (k, v) in zip(self.model.model.layers, caches):
            cache = layer.self_attn.attn.kv_cache[0].reshape(2, 12, 2, 4)
            torch.testing.assert_close(cache[0, self.slots], k, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(cache[1, self.slots], v, atol=2e-6, rtol=2e-5)
            self.assertEqual(cache[:, unused].abs().sum(), 0)
        decoded = self.model(self.ids[-1:], self.positions[-1:] + 1)
        torch.testing.assert_close(decoded, self.model.model.embed_tokens(self.ids[-1:]))
        metrics = self.worker.repair_metrics()
        self.assertEqual(metrics["repair_prefill_calls"], 1)
        self.assertEqual(metrics["native_forward_calls"], 1)
        self.assertFalse(metrics["pending"])

    def test_invalid_metadata_fails_before_cache_write_and_disarms(self):
        self.arm()
        self.meta.slot_mapping = self.slots.clone()
        self.meta.slot_mapping[0] = -1
        with self.assertRaisesRegex(ValueError, "padding"):
            self.model(self.ids, self.positions)
        self.assertEqual(self.kernel_calls, [])
        self.assertFalse(self.worker.repair_metrics()["pending"])

    def test_prompt_logprobs_fail_before_cache_write(self):
        self.arm()
        self.worker.model_runner.requests["request"].sampling_params.prompt_logprobs = 0
        with self.assertRaisesRegex(ValueError, "logprobs"):
            self.model(self.ids, self.positions)
        self.assertEqual(self.kernel_calls, [])
        self.assertFalse(self.worker.repair_metrics()["pending"])

    def test_paged_writer_rejects_duplicate_layer(self):
        writer, written = _make_kv_writer(self.model, len(self.ids))
        writer(0, torch.zeros(9, 2, 4), torch.zeros(9, 2, 4))
        self.assertEqual(written, {0})
        with self.assertRaisesRegex(ValueError, "exactly once"):
            writer(0, torch.zeros(9, 2, 4), torch.zeros(9, 2, 4))

    def test_incompatible_engine_rejected(self):
        self.worker.vllm_config.cache_config.enable_prefix_caching = True
        with self.assertRaisesRegex(ValueError, "prefix caching"):
            self.worker.repair_initialize()
        self.assertFalse(hasattr(self.worker, "_repair_runtime"))


class RunnerTest(unittest.TestCase):
    def test_same_segment_ids_used_without_implicit_special_tokens(self):
        tokenizer = SimpleNamespace(encode=mock.Mock(side_effect=[[1, 2], [3, 4], [5]]))
        result = tokenize_request(tokenizer, "prefix", ["doc"], "suffix")
        self.assertEqual(result, ([1, 2], [[3, 4]], [5]))
        self.assertEqual(tokenizer.encode.call_args_list,
                         [mock.call(text, add_special_tokens=False) for text in ("prefix", "doc", "suffix")])

    def test_ttft_is_first_emitted_token_and_not_total_generation(self):
        class Engine:
            remaining = 0

            def has_unfinished_requests(self):
                return self.remaining > 0

            def add_request(self, request_id, prompt, sampling):
                self.request_id = request_id
                self.prompt = prompt
                self.remaining = 3

            def step(self):
                self.remaining -= 1
                if self.remaining == 2:
                    return []
                completion = SimpleNamespace(token_ids=[17] if self.remaining else [17, 18],
                                             text="answer", finish_reason="length")
                return [SimpleNamespace(request_id=self.request_id, outputs=[completion],
                                        finished=self.remaining == 0)]

        engine = Engine()
        with mock.patch("vllm_repair.run.time.perf_counter", side_effect=[100., 102., 108.]):
            result = measure_request(SimpleNamespace(llm_engine=engine), [1, 2, 3], object())
        self.assertEqual(result["ttft_s"], 2.)
        self.assertEqual(result["generation_wall_s"], 8.)
        self.assertEqual(result["token_ids"], [17, 18])
        self.assertEqual(engine.prompt, {"prompt_token_ids": [1, 2, 3]})


if __name__ == "__main__":
    unittest.main()
