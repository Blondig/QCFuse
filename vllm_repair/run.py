"""Run one JSON RAG request with native vLLM 0.9.2 selective prefill.

Usage: python -m vllm_repair.run --model MODEL --input request.json
"""

import argparse
import json
import math
import os
from pathlib import Path
import time
import uuid


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path, help="JSON with prefix, documents, and suffix (or query)")
    parser.add_argument("--method", choices=("prophet", "prophet_fo", "prophet_fo_residual", "prophet_fo_mixed"), default="prophet_fo")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--probe-layer", type=int, default=2, help="Zero-based shallow drift layer; Prophet scans all layers")
    parser.add_argument("--influence-weight", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--cache-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--attention-block-size", type=int, default=128)
    parser.add_argument("--compare-full", action="store_true", help="Also run native full prefill with identical token IDs")
    parser.add_argument("--dtype", choices=("auto", "half", "bfloat16"), default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6, help="Leave GPU space for the repair cache and temporary attention tensors")
    parser.add_argument("--max-model-len", type=int, default=None, help="Defaults to prompt tokens plus max-tokens")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_request(path):
    with path.open(encoding="utf-8") as stream:
        request = json.load(stream)
    if not isinstance(request, dict):
        raise ValueError("The input JSON must be an object")
    prefix = request.get("prefix")
    documents = request.get("documents")
    if "suffix" in request and "query" in request:
        raise ValueError("Provide either suffix or query, not both")
    suffix = request.get("suffix", request.get("query"))
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be a nonempty string")
    if not isinstance(documents, list) or not documents or any(not isinstance(doc, str) or not doc for doc in documents):
        raise ValueError("documents must be a nonempty list of nonempty strings")
    if not isinstance(suffix, str) or not suffix:
        raise ValueError("suffix (or query) must be a nonempty string")
    return prefix, documents, suffix


def tokenize_request(tokenizer, prefix, documents, suffix):
    # No separators, BOS, or chat template are inserted implicitly. Include all
    # required delimiters in the JSON strings. The full baseline uses these IDs.
    chunks = [tokenizer.encode(text, add_special_tokens=False) for text in (prefix, *documents, suffix)]
    if any(not chunk for chunk in chunks):
        raise ValueError("Every prefix, document, and suffix must tokenize to at least one token")
    return chunks[0], chunks[1:-1], chunks[-1]


def _rpc(llm, method, **kwargs):
    results = llm.collective_rpc(method, kwargs=kwargs)
    if len(results) != 1:
        raise RuntimeError("Expected exactly one worker RPC result")
    return results[0]


def measure_request(llm, prompt_ids, sampling_params):
    """Measure submission-to-first-observed-token latency via V1 engine.step.

    This includes scheduling, model execution, IPC, and output processing. It is
    not a GPU kernel duration and does not use total generation wall time as TTFT.
    """
    engine = llm.llm_engine
    if engine.has_unfinished_requests():
        raise RuntimeError("The runner requires an idle engine before submitting a request")
    request_id = "repair-" + uuid.uuid4().hex
    started = time.perf_counter()
    first_token_at = None
    final_output = None
    try:
        engine.add_request(request_id, {"prompt_token_ids": prompt_ids}, sampling_params)
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.request_id != request_id:
                    raise RuntimeError("The single-request engine returned an unexpected request")
                if first_token_at is None and any(item.token_ids for item in output.outputs):
                    first_token_at = time.perf_counter()
                if output.finished:
                    final_output = output
        ended = time.perf_counter()
    except BaseException:
        try:
            engine.abort_request([request_id])
        except Exception:
            pass
        raise
    if final_output is None or not final_output.outputs or first_token_at is None:
        raise RuntimeError("The request finished without an observable generated token")
    completion = final_output.outputs[0]
    return {
        "text": completion.text, "token_ids": list(completion.token_ids),
        "ttft_s": first_token_at - started, "generation_wall_s": ended - started,
        "ttft_measurement": "V1 engine submission to first returned output containing a token",
        "finish_reason": completion.finish_reason,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("ratio", "influence_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.probe_layer < (0 if args.method == "prophet" else 1):
        parser.error("--probe-layer must be >= 1 for FO, or >= 0 for Prophet")
    if args.max_tokens < 1 or args.attention_block_size < 1:
        parser.error("--max-tokens and --attention-block-size must be positive")
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    prefix, documents, suffix = load_request(args.input)

    # Set these before importing either the vLLM engine or its worker modules.
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    if vllm.__version__.split("+")[0] != "0.9.2":
        raise RuntimeError(f"This runner requires vLLM 0.9.2; found {vllm.__version__}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prefix_ids, document_ids, suffix_ids = tokenize_request(tokenizer, prefix, documents, suffix)
    prompt_ids = prefix_ids + [token for doc in document_ids for token in doc] + suffix_ids
    required_length = len(prompt_ids) + args.max_tokens
    max_model_len = args.max_model_len if args.max_model_len is not None else required_length
    if max_model_len < required_length:
        parser.error("--max-model-len must cover the prompt and --max-tokens")
    llm = LLM(
        model=args.model, dtype=args.dtype, seed=args.seed,
        tensor_parallel_size=1, pipeline_parallel_size=1, max_num_seqs=1,
        max_model_len=max_model_len, max_num_batched_tokens=max_model_len,
        enable_prefix_caching=False, enable_chunked_prefill=False,
        enforce_eager=True, compilation_config=0,
        gpu_memory_utilization=args.gpu_memory_utilization,
        worker_extension_cls="vllm_repair.worker.RepairWorkerExtension",
    )
    initialized = _rpc(llm, "repair_initialize", cache_device=args.cache_device,
                       attention_block_size=args.attention_block_size)
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens,
                              seed=args.seed, output_kind=RequestOutputKind.CUMULATIVE)
    report = {
        "engine": initialized,
        "config": {"method": args.method, "ratio": args.ratio, "probe_layer": args.probe_layer,
                   "influence_weight": args.influence_weight},
        "tokens": {"prefix": len(prefix_ids), "documents": [len(doc) for doc in document_ids],
                   "suffix": len(suffix_ids), "prompt": len(prompt_ids)},
        "timing_note": "Single-request diagnostic timings; offline cache preparation is excluded from TTFT. No steady-state speedup claim.",
    }
    if args.compare_full:
        report["full"] = measure_request(llm, prompt_ids, sampling)
    prepare_started = time.perf_counter()
    report["offline_prepare"] = _rpc(llm, "repair_prepare_chunks", chunks=[prefix_ids, *document_ids])
    report["offline_prepare"]["rpc_wall_s"] = time.perf_counter() - prepare_started
    _rpc(llm, "repair_arm", prefix_ids=prefix_ids, documents=document_ids,
         suffix_ids=suffix_ids, config=report["config"])
    try:
        report["repair"] = measure_request(llm, prompt_ids, sampling)
        report["repair"]["runtime"] = _rpc(llm, "repair_metrics")
    except BaseException:
        # Preserve a model/engine failure if its process also fails to answer
        # the cleanup RPC. The worker itself disarms in a finally block.
        try:
            _rpc(llm, "repair_disarm")
        except Exception:
            pass
        raise
    else:
        _rpc(llm, "repair_disarm")
    if report["repair"]["runtime"]["repair_prefill_calls"] != 1:
        raise RuntimeError("The request did not execute exactly one repair prefill")
    if args.compare_full:
        report["same_generated_tokens"] = report["full"]["token_ids"] == report["repair"]["token_ids"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
