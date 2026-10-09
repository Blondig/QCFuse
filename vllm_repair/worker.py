"""vLLM 0.9.2 V1 worker extension for selective prefill and native decode.

Importing this module does not import vLLM or torch. The extension is initialized
by collective_rpc after vLLM has loaded the model and allocated its paged cache.
"""

from types import MethodType


def _validate_engine(worker, model):
    import vllm
    from vllm import envs

    if vllm.__version__.split("+")[0] != "0.9.2" or not envs.VLLM_USE_V1:
        raise RuntimeError("Selective repair requires vLLM 0.9.2 with VLLM_USE_V1=1")
    cfg = worker.vllm_config
    parallel = cfg.parallel_config
    if any(getattr(parallel, name, 1) != 1 for name in (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"
    )):
        raise ValueError("Selective repair requires TP=PP=DP=1")
    scheduler = cfg.scheduler_config
    if scheduler.max_num_seqs != 1:
        raise ValueError("Selective repair requires max_num_seqs=1")
    if getattr(scheduler, "enable_chunked_prefill", False) or getattr(
        scheduler, "chunked_prefill_enabled", False
    ):
        raise ValueError("Disable chunked prefill for selective repair")
    if cfg.cache_config.enable_prefix_caching:
        raise ValueError("Disable automatic prefix caching for selective repair")
    if not cfg.model_config.enforce_eager or cfg.compilation_config.level != 0:
        raise ValueError("Selective repair requires enforce_eager=True and compilation_config=0")
    if cfg.model_config.quantization or getattr(cfg, "quant_config", None):
        raise ValueError("Quantized model weights are unsupported")
    if cfg.cache_config.cache_dtype != "auto":
        raise ValueError("Selective repair requires kv_cache_dtype='auto'")
    if getattr(cfg.cache_config, "calculate_kv_scales", False):
        raise ValueError("Dynamic KV quantization scales are unsupported")
    if getattr(cfg.cache_config, "cpu_offload_gb", 0):
        raise ValueError("CPU weight offloading is unsupported by the layerwise runtime")
    for name in ("lora_config", "speculative_config", "kv_transfer_config", "prompt_adapter_config"):
        if getattr(cfg, name, None) is not None:
            raise ValueError(f"Selective repair does not support {name}")
    if type(model).__name__ not in {"LlamaForCausalLM", "Qwen3ForCausalLM"}:
        raise ValueError("Only native dense LlamaForCausalLM and Qwen3ForCausalLM are supported")
    if not type(model).__module__.startswith("vllm.model_executor.models."):
        raise ValueError("The worker must use the native vLLM model implementation")
    if getattr(worker.model_runner, "use_aux_hidden_state_outputs", False):
        raise ValueError("Auxiliary hidden-state outputs are unsupported")
    if not getattr(cfg.model_config.hf_config, "is_causal", True):
        raise ValueError("Bidirectional attention is unsupported")
    for layer in model.model.layers:
        attention = layer.self_attn.attn
        impl = attention.impl
        if type(impl).__module__ != "vllm.v1.attention.backends.flash_attn":
            raise ValueError("Selective repair requires the V1 FLASH_ATTN backend")
        if attention.attn_type != "decoder":
            raise ValueError("Only causal decoder attention is supported")
        if attention.sliding_window is not None or tuple(impl.sliding_window) != (-1, -1):
            raise ValueError("Sliding-window attention is unsupported")
        if getattr(impl, "alibi_slopes", None) is not None:
            raise ValueError("ALiBi attention is unsupported")
        if getattr(impl, "logits_soft_cap", 0) not in (None, 0):
            raise ValueError("Attention logits soft-capping is unsupported")
        if getattr(impl, "kv_sharing_target_layer_name", None) is not None:
            raise ValueError("Cross-layer KV sharing is unsupported")
        if getattr(impl, "use_irope", False):
            raise ValueError("Interleaved no-RoPE attention is unsupported")


def _validate_request(worker, token_count):
    batch = worker.model_runner.input_batch
    request_ids = list(batch.req_id_to_index)
    if len(request_ids) != 1:
        raise ValueError("Repair prefill must contain exactly one request")
    request = worker.model_runner.requests[request_ids[0]]
    if request.num_computed_tokens != 0 or len(request.prompt_token_ids) != token_count:
        raise ValueError("Repair requires a complete, uncached prompt starting at position zero")
    if request.output_token_ids:
        raise ValueError("Repair cannot be armed for an existing decode request")
    if request.sampling_params is None:
        raise ValueError("Repair requires a text generation request")
    if request.sampling_params.prompt_logprobs is not None:
        raise ValueError("Prompt logprobs are unsupported because inactive hidden rows are omitted")
    if request.mm_inputs or request.lora_request is not None:
        raise ValueError("Multimodal inputs and LoRA requests are unsupported")


def _make_kv_writer(model, token_count):
    """Validate the native full-prefill mapping before writing any layer."""
    import torch
    from vllm.attention.utils.fa_utils import reshape_and_cache_flash
    from vllm.forward_context import get_forward_context

    context = get_forward_context()
    metadata = context.attn_metadata
    mappings = []
    validated = {}
    for layer in model.model.layers:
        attention = layer.self_attn.attn
        meta = metadata[attention.layer_name] if isinstance(metadata, dict) else metadata
        if meta is None:
            raise ValueError("Repair cannot run during a profiling or dummy forward")
        if id(meta) not in validated:
            if meta.num_actual_tokens != token_count:
                raise ValueError("Repair metadata must cover the entire prompt")
            if meta.query_start_loc.tolist() != [0, token_count]:
                raise ValueError("Repair requires one unchunked query sequence")
            if meta.seq_lens.tolist() != [token_count]:
                raise ValueError("Repair metadata must have no computed prefix")
            if getattr(meta, "use_cascade", False):
                raise ValueError("Cascade attention is unsupported")
            slots = meta.slot_mapping
            if slots.ndim != 1 or slots.numel() != token_count or slots.dtype != torch.int64:
                raise ValueError("Unexpected paged-cache slot mapping")
            lowest, highest = int(slots.min().item()), int(slots.max().item())
            if lowest < 0:
                raise ValueError("Repair cannot write padding slots")
            validated[id(meta)] = (slots, highest)
        slots, highest = validated[id(meta)]
        cache = attention.kv_cache[context.virtual_engine]
        if cache.ndim != 5 or cache.shape[0] != 2:
            raise ValueError("Unexpected FLASH_ATTN paged-cache shape")
        if highest >= cache.shape[1] * cache.shape[2]:
            raise ValueError("Paged-cache slot index is out of range")
        mappings.append((attention, cache, slots))

    written = set()

    def write_kv(layer_index, key, value):
        if layer_index in written or not 0 <= layer_index < len(mappings):
            raise ValueError("Each decoder layer must write KV exactly once")
        attention, cache, slots = mappings[layer_index]
        expected = (token_count, attention.num_kv_heads, attention.head_size)
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(f"Repair KV must have shape {expected}")
        if key.device != cache.device or value.device != cache.device or slots.device != cache.device:
            raise ValueError("Repair KV and slot mapping must be on the paged-cache device")
        if key.dtype != cache.dtype or value.dtype != cache.dtype:
            raise ValueError("Repair KV dtype must match the unquantized paged cache")
        key_cache, value_cache = cache.unbind(0)
        reshape_and_cache_flash(
            key.contiguous(), value.contiguous(), key_cache, value_cache, slots,
            attention.impl.kv_cache_dtype, attention._k_scale, attention._v_scale,
        )
        written.add(layer_index)

    return write_kv, written


class RepairWorkerExtension:
    """Mixin installed using vLLM's worker_extension_cls setting."""

    def repair_initialize(self, cache_device="cpu", attention_block_size=128):
        from .runtime import SparseRepairRuntime

        if hasattr(self, "_repair_runtime"):
            raise RuntimeError("The repair worker is already initialized")
        model = self.model_runner.model
        _validate_engine(self, model)
        runtime = SparseRepairRuntime(model, cache_device, attention_block_size)
        original_forward = model.forward
        worker = self

        def forward(_model, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
            if runtime.pending is None:
                worker._repair_native_forward_calls += 1
                return original_forward(
                    input_ids=input_ids, positions=positions,
                    intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds, **kwargs,
                )
            try:
                if intermediate_tensors is not None or inputs_embeds is not None or kwargs:
                    raise ValueError("Repair supports only ordinary token-ID prefill inputs")
                if input_ids is None or input_ids.ndim != 1 or input_ids.numel() == 0:
                    raise ValueError("Repair requires nonempty one-dimensional token IDs")
                token_count = input_ids.numel()
                _validate_request(worker, token_count)
                write_kv, written = _make_kv_writer(model, token_count)
                hidden = runtime.run_prefill(input_ids, positions, write_kv)
                if len(written) != len(model.model.layers):
                    raise RuntimeError("Repair did not populate every native decoder KV cache")
                if hidden.ndim != 2 or hidden.shape[0] != token_count:
                    raise RuntimeError("Repair must return full prompt-shaped final hidden states")
                worker._repair_prefill_calls += 1
                return hidden
            finally:
                runtime.disarm()

        self._repair_runtime = runtime
        self._repair_native_forward_calls = 0
        self._repair_prefill_calls = 0
        self._repair_original_forward = original_forward
        model.forward = MethodType(forward, model)
        return {
            "vllm_version": "0.9.2", "model_class": type(model).__name__,
            "decoder_layers": len(model.model.layers), "cache_device": str(runtime.cache_device),
            "attention_block_size": runtime.attention_block_size,
        }

    def repair_prepare_chunks(self, chunks):
        return self._repair_runtime.prepare_chunks(chunks)

    def repair_arm(self, prefix_ids, documents, suffix_ids, config=None):
        from .runtime import RepairConfig

        if not prefix_ids or not suffix_ids:
            raise ValueError("Both prefix_ids and suffix_ids must be nonempty")
        self._repair_runtime.arm(prefix_ids, documents, suffix_ids, RepairConfig(**(config or {})))
        self._repair_native_forward_calls = 0
        self._repair_prefill_calls = 0
        return {"armed": True, "prompt_tokens": len(self._repair_runtime.pending.token_ids)}

    def repair_metrics(self):
        return {
            **self._repair_runtime.metrics,
            "repair_prefill_calls": self._repair_prefill_calls,
            "native_forward_calls": self._repair_native_forward_calls,
            "pending": self._repair_runtime.pending is not None,
        }

    def repair_disarm(self):
        self._repair_runtime.disarm()

    def repair_clear(self):
        self._repair_runtime.clear()
