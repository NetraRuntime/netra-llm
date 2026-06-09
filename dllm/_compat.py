"""Compatibility shims so dllm imports on bleeding-edge ``transformers``.

Qwen3.5 (``qwen3_5``) only exists in very recent ``transformers`` (git main), but
that line dropped a couple of symbols that dllm's pinned ``peft`` and the ``llada2``
pipeline still import at module-load time. Importing this module **first** (it is the
first import in ``dllm/__init__.py``) restores those names before anything else
touches ``transformers``, so ``import dllm`` works without pinning older transformers.

Each shim is a no-op when the symbol already exists, and is wrapped defensively so a
shim can never itself break the import.
"""

try:
    import transformers

    # peft does ``from transformers import HybridCache``; transformers>=5 removed it.
    if not hasattr(transformers, "HybridCache"):
        try:
            from transformers.cache_utils import DynamicCache as _CacheFallback
        except Exception:  # pragma: no cover - extremely defensive
            _CacheFallback = object
        transformers.HybridCache = _CacheFallback

    # dllm.pipelines.llada2 does ``from transformers.utils.import_utils import
    # is_torch_fx_available``; transformers>=5 removed it.
    try:
        import transformers.utils.import_utils as _import_utils

        if not hasattr(_import_utils, "is_torch_fx_available"):
            _import_utils.is_torch_fx_available = lambda: False
            # also expose on the parent namespace some code imports from
            try:
                transformers.utils.is_torch_fx_available = (
                    _import_utils.is_torch_fx_available
                )
            except Exception:  # pragma: no cover
                pass
    except Exception:  # pragma: no cover
        pass
except Exception:  # pragma: no cover - never let the shim block importing dllm
    pass
