import torch, transformers
from dllm.tools.extract_qwen3_5_text import extract_text_backbone


def test_extract_produces_causal_lm_with_mask_token(tmp_path):
    # Build a tiny multimodal model from the real config shrunk down, save, extract.
    cfg = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
    cfg.text_config.num_hidden_layers = 4         # 3 linear + 1 full (interval 4)
    # layer_types is a stale length-24 list on the real config; shrink it to match
    # num_hidden_layers so the config validator accepts the tiny model.
    cfg.text_config.layer_types = [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ]
    cfg.text_config.hidden_size = 128
    cfg.text_config.head_dim = 32
    cfg.text_config.intermediate_size = 256
    src_dir = tmp_path / "mm"
    mm = transformers.Qwen3_5ForConditionalGeneration(cfg)
    mm.save_pretrained(src_dir)
    transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base").save_pretrained(src_dir)

    out_dir = tmp_path / "text"
    extract_text_backbone(str(src_dir), str(out_dir), dtype="float32")

    tok = transformers.AutoTokenizer.from_pretrained(out_dir)
    assert tok.convert_tokens_to_ids("<|mask|>") != tok.unk_token_id
    model = transformers.AutoModelForCausalLM.from_pretrained(out_dir)
    assert model.config.model_type == "qwen3_5_text"
    assert model.get_input_embeddings().weight.shape[0] == len(tok)
    # weights copied (not random): language_model embed row matches
    assert torch.allclose(
        model.get_input_embeddings().weight[:10],
        mm.model.language_model.embed_tokens.weight[:10].to(model.dtype),
        atol=1e-4,
    )
