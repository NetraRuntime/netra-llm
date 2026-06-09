def test_transformers_has_qwen3_5():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: F401


def test_dllm_imports_with_this_transformers():
    import dllm  # noqa: F401
    import dllm.pipelines.a2d  # noqa: F401
