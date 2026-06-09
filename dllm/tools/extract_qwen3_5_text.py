"""
Extract the text backbone of a multimodal Qwen3.5 checkpoint into a text-only
Qwen3_5ForCausalLM, add a <|mask|> token, and resize embeddings.

Run:
    python dllm/tools/extract_qwen3_5_text.py \
        --model_name_or_path "Qwen/Qwen3.5-0.8B-Base" \
        --output_dir ".models/qwen3_5-0.8b-text"
"""
from dataclasses import dataclass

import transformers
import tyro

import dllm


def extract_text_backbone(model_name_or_path: str, output_dir: str, dtype: str = "bfloat16"):
    mm = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name_or_path, dtype=dtype
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    text = transformers.Qwen3_5ForCausalLM(mm.config.text_config)
    # The multimodal text backbone lives at mm.model.language_model (a Qwen3_5TextModel)
    missing, unexpected = text.model.load_state_dict(
        mm.model.language_model.state_dict(), strict=False
    )
    print("text.model missing:", missing)
    print("text.model unexpected:", unexpected)
    text.tie_weights()  # lm_head <- model.embed_tokens (tie_word_embeddings=True)

    # Add the absorbing mask token and grow embeddings by 1 (no reserved <|mask|> in vocab)
    added = tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
    if added:
        text.resize_token_embeddings(len(tokenizer))

    text.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


@dataclass
class ScriptArguments:
    model_name_or_path: str = "Qwen/Qwen3.5-0.8B-Base"
    output_dir: str = ".models/qwen3_5-0.8b-text"
    dtype: str = "bfloat16"

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


def main():
    args = tyro.cli(ScriptArguments)
    extract_text_backbone(args.model_name_or_path, args.output_dir, args.dtype)


if __name__ == "__main__":
    main()
