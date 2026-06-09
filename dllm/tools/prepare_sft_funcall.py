"""
Prepare a unified function-calling SFT corpus (pre-tokenized, assistant-only loss).

Sources
-------
- NousResearch/hermes-function-calling-v1 (public): multi-turn agentic traces in ShareGPT
  format with <tool_call>/<tool_response> tags already in the text.
- Salesforce/xlam-function-calling-60k (GATED — needs HF_TOKEN): single-turn
  query/tools/answers JSON.
- jaeyong2/Id-functioncall (public): single-turn Indonesian context/question/functions/
  function_call.

Every row is converted to chat `messages`, rendered with the model's chat template, and
tokenized with MULTI-TURN ASSISTANT-ONLY labels: loss (and diffusion noising — the BD3LM
trainer masks only labels != -100) covers every assistant turn's content + end-of-turn
token; system/user/tool turns and assistant headers stay -100 (clean conditioning).

Output: save_to_disk DatasetDict {train, test} of {input_ids, labels} at --output_dir.
Train with: examples/a2d/bd3lm/sft.py --dataset_args <output_dir> --load_preprocessed_data True

Usage:
    python dllm/tools/prepare_sft_funcall.py \
        --tokenizer_path /data/a2d/qwen3_5-4b --output_dir /data/datasets/sft-funcall-4096 \
        --max_length 4096 --num_proc 32
"""

import argparse
import json
import os
import re

import datasets
import transformers

# Hermes-style canonical system preamble (EN) and an Indonesian twin, both using the same
# <tools>/<tool_call> tag protocol so the trigger format is consistent across languages.
SYSTEM_EN = (
    "You are a function calling AI model. You are provided with function signatures "
    "within <tools> </tools> XML tags. You may call one or more functions to assist with "
    "the user query. Don't make assumptions about what values to plug into functions.\n"
    "<tools>\n{tools}\n</tools>\n"
    "For each function call return a json object with function name and arguments within "
    "<tool_call> </tool_call> tags."
)
SYSTEM_ID = (
    "Anda adalah model AI pemanggil fungsi. Anda diberikan signature fungsi di dalam tag "
    "XML <tools> </tools>. Anda dapat memanggil satu atau beberapa fungsi untuk membantu "
    "pertanyaan pengguna. Jangan berasumsi tentang nilai argumen fungsi.\n"
    "<tools>\n{tools}\n</tools>\n"
    "Untuk setiap pemanggilan fungsi, kembalikan objek json berisi nama fungsi dan "
    "argumen di dalam tag <tool_call> </tool_call>."
)

_TOOL_RESP_RE = re.compile(r"^\s*<tool_response>\s*(.*?)\s*</tool_response>\s*$", re.S)


def hermes_to_messages(row):
    """ShareGPT from/value -> messages. Tool turns lose their <tool_response> wrapper
    (the chat template re-adds its own), assistant <tool_call> text passes through."""
    role_map = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
    messages = []
    for turn in row["conversations"]:
        role = role_map.get(turn["from"])
        if role is None:
            return None
        value = turn["value"]
        if role == "tool":
            m = _TOOL_RESP_RE.match(value)
            if m:
                value = m.group(1)
        messages.append({"role": role, "content": value})
    return messages


def xlam_to_messages(row):
    """query/tools/answers (JSON strings) -> single-turn tool-call exchange."""
    try:
        tools = json.dumps(json.loads(row["tools"]), ensure_ascii=False)
        answers = json.loads(row["answers"])
    except (json.JSONDecodeError, TypeError):
        return None
    calls = "\n".join(
        f"<tool_call>\n{json.dumps(c, ensure_ascii=False)}\n</tool_call>" for c in answers
    )
    return [
        {"role": "system", "content": SYSTEM_EN.format(tools=tools)},
        {"role": "user", "content": row["query"]},
        {"role": "assistant", "content": calls},
    ]


def idfc_to_messages(row):
    """Indonesian context/question/functions/function_call -> single-turn exchange with
    the Indonesian system preamble (same tag protocol)."""
    try:
        tools = json.dumps(json.loads(row["functions"]), ensure_ascii=False)
        call = json.dumps(json.loads(row["function_call"]), ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return None
    user = (row.get("context") or "").strip()
    question = (row.get("question") or "").strip()
    user = f"{user}\n\n{question}" if user else question
    if not user:
        return None
    return [
        {"role": "system", "content": SYSTEM_ID.format(tools=tools)},
        {"role": "user", "content": user},
        {"role": "assistant", "content": f"<tool_call>\n{call}\n</tool_call>"},
    ]


def _template_ids(tokenizer, messages, add_generation_prompt):
    """apply_chat_template -> flat token list across transformers versions
    (v5 returns a BatchEncoding, v4 a list)."""
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    return out["input_ids"] if hasattr(out, "keys") else out


def kolosal_bench_to_rows(path):
    """Kolosal benchmark-eval dump -> single-turn rows. Targets are the GOLD
    expected_answer (not the benchmarked model's output). Summarize/translate/
    paraphrase over Indonesian customer-service conversations."""
    with open(path) as f:
        data = json.load(f)
    rows = []
    for conv in data.get("conversations", []):
        for chat in conv.get("chats", []):
            inp = chat.get("input") or {}
            q, a = (inp.get("question") or "").strip(), (inp.get("expected_answer") or "").strip()
            if chat.get("status") != "success" or not q or not a:
                continue
            rows.append(
                {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
            )
    return rows


def multiturn_sft_map_fn(row, *, tokenizer, max_length):
    """Tokenize messages with assistant-only labels across ALL turns.

    ChatML-family templates are prefix-stable (each turn appends a self-contained
    <|im_start|>...<|im_end|> segment), so per-turn token spans are computed from
    incremental prefix lengths. The assistant *header* (everything add_generation_prompt
    would append, including any <think> scaffold) stays masked; content + end-of-turn
    supervise.
    """
    messages = row["messages"]
    full = _template_ids(tokenizer, messages, False)
    if len(full) > max_length:
        return {"input_ids": [], "labels": []}
    labels = [-100] * len(full)
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        upto_prev = _template_ids(tokenizer, messages[:i], True)  # prefix + assistant header
        upto_here = _template_ids(tokenizer, messages[: i + 1], False)
        start, end = len(upto_prev), len(upto_here)
        # hard prefix-stability verification (drop the row if the template misbehaves)
        if not (0 < start < end <= len(full)) or full[:end] != upto_here:
            return {"input_ids": [], "labels": []}
        labels[start:end] = full[start:end]
    if all(l == -100 for l in labels):
        return {"input_ids": [], "labels": []}
    return {"input_ids": full, "labels": labels}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--num_proc", type=int, default=32)
    p.add_argument("--test_size", type=int, default=500)
    p.add_argument("--xlam_cap", type=int, default=30000)
    p.add_argument("--idfc_cap", type=int, default=0, help="cap Id-functioncall rows (0 = all)")
    p.add_argument("--extra_json", default="", help="Kolosal benchmark-eval JSON path (optional)")
    args = p.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
    assert tok.chat_template, "tokenizer has no chat template"

    parts = []

    # --- Hermes (all configs are agentic/structured-output relevant) ---
    for cfg in (
        "func_calling",
        "func_calling_singleturn",
        "glaive_func_calling",
        "json_mode_agentic",
        "json_mode_singleturn",
    ):
        ds = datasets.load_dataset("NousResearch/hermes-function-calling-v1", cfg, split="train")
        ds = ds.map(
            lambda r: {"messages": hermes_to_messages(r)},
            remove_columns=ds.column_names,
            num_proc=args.num_proc,
            desc=f"hermes:{cfg}",
        ).filter(lambda r: r["messages"] is not None, num_proc=args.num_proc)
        parts.append(ds)
        print(f"[sft-prep] hermes/{cfg}: {len(ds):,} rows", flush=True)

    # --- xlam (gated; best-effort) ---
    if os.environ.get("HF_TOKEN"):
        try:
            ds = datasets.load_dataset(
                "Salesforce/xlam-function-calling-60k", split="train", token=os.environ["HF_TOKEN"]
            )
            if len(ds) > args.xlam_cap:
                ds = ds.shuffle(seed=42).select(range(args.xlam_cap))
            ds = ds.map(
                lambda r: {"messages": xlam_to_messages(r)},
                remove_columns=ds.column_names,
                num_proc=args.num_proc,
                desc="xlam",
            ).filter(lambda r: r["messages"] is not None, num_proc=args.num_proc)
            parts.append(ds)
            print(f"[sft-prep] xlam: {len(ds):,} rows", flush=True)
        except Exception as e:
            print(f"[sft-prep] WARN xlam skipped: {type(e).__name__}: {e}", flush=True)
    else:
        print("[sft-prep] WARN xlam SKIPPED (no HF_TOKEN; gated dataset)", flush=True)

    # --- Indonesian function calling ---
    ds = datasets.load_dataset("jaeyong2/Id-functioncall", split="train")
    if args.idfc_cap and len(ds) > args.idfc_cap:
        ds = ds.shuffle(seed=42).select(range(args.idfc_cap))
    ds = ds.map(
        lambda r: {"messages": idfc_to_messages(r)},
        remove_columns=ds.column_names,
        num_proc=args.num_proc,
        desc="id-funcall",
    ).filter(lambda r: r["messages"] is not None, num_proc=args.num_proc)
    parts.append(ds)
    print(f"[sft-prep] id-functioncall: {len(ds):,} rows", flush=True)

    # --- Kolosal benchmark gold (summarize/translate/paraphrase, ID customer-service) ---
    if args.extra_json and os.path.exists(args.extra_json):
        rows = kolosal_bench_to_rows(args.extra_json)
        parts.append(datasets.Dataset.from_list(rows))
        print(f"[sft-prep] kolosal-bench: {len(rows):,} rows", flush=True)
    elif args.extra_json:
        print(f"[sft-prep] WARN extra_json not found: {args.extra_json}", flush=True)

    merged = datasets.concatenate_datasets(parts).shuffle(seed=42)
    print(f"[sft-prep] merged: {len(merged):,} rows; tokenizing...", flush=True)

    tokenized = merged.map(
        multiturn_sft_map_fn,
        fn_kwargs={"tokenizer": tok, "max_length": args.max_length},
        remove_columns=merged.column_names,
        num_proc=args.num_proc,
        desc="tokenize+mask",
    ).filter(lambda r: len(r["input_ids"]) > 0, num_proc=args.num_proc)

    n_test = min(args.test_size, len(tokenized) // 20)
    split = tokenized.train_test_split(test_size=n_test, seed=42)
    split.save_to_disk(args.output_dir)
    n_tok = sum(len(x) for x in tokenized[:1000]["input_ids"]) / min(1000, len(tokenized))
    print(
        f"[sft-prep] DONE train={len(split['train']):,} test={len(split['test']):,} "
        f"avg_len~{n_tok:.0f} -> {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
