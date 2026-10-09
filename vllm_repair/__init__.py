"""Independent vLLM repair utilities, imported lazily for CPU-only CLI help."""

from importlib import import_module

__all__ = [
    "causal_attention",
    "compute_attention_influence",
    "fuse_scores",
    "mean_attention_scores",
    "percentile_rank",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".attention" if name == "causal_attention" else ".scoring"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
