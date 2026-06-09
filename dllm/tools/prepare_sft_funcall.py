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


def bactrian_to_messages(row):
    """Bactrian-X {instruction, input, output} -> single-turn chat (Indonesian config)."""
    instr = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    out = (row.get("output") or "").strip()
    if not instr or not out:
        return None
    user = f"{instr}\n\n{inp}" if inp else instr
    return [{"role": "user", "content": user}, {"role": "assistant", "content": out}]


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


# assistant turn in a ChatML render: header (+ optional <think> scaffold) then content
# up to <|im_end|>. The Qwen template renders the <think> scaffold only on the final
# assistant turn, so prefix-based span math is NOT stable — we instead locate spans in
# the single full render and map char offsets -> token indices.
_ASSISTANT_BLOCK_RE = re.compile(
    r"<\|im_start\|>assistant\n(?:<think>.*?</think>\n*)?(.*?<\|im_end\|>)", re.S
)


def _tokenize_with_spans(messages, tokenizer, supervise):
    """Tokenize a conversation with assistant-only labels on the message indices in
    `supervise` (other assistant turns act as pure context). Spans are located in the
    one full template render via regex + fast-tokenizer offset mapping — template-
    agnostic, no prefix re-renders. The assistant header and any <think> scaffold stay
    masked; content + end-of-turn token supervise. Returns (ids, labels) or None."""
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return None  # template refuses this window (e.g. no user turn)
    blocks = list(_ASSISTANT_BLOCK_RE.finditer(text))
    a_msg_idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if len(blocks) != len(a_msg_idx):
        return None  # render doesn't match message structure; drop
    char_spans = [
        blocks[k].span(1) for k, mi in enumerate(a_msg_idx) if mi in supervise
    ]
    if not char_spans:
        return None
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    labels = [-100] * len(ids)
    for cs, ce in char_spans:
        for t, (ts, te) in enumerate(offsets):
            if ts >= cs and te <= ce and ts < te:
                labels[t] = ids[t]
    if all(l == -100 for l in labels):
        return None
    return ids, labels


def _shrink_user_middle(messages, tokenizer, max_length, min_keep_chars=512):
    """For a minimal window that still exceeds max_length: middle-truncate its longest
    user turn (instructions tend to sit at the top, the question at the bottom, so keep
    both ends). Returns a fitted messages list or None."""
    msgs = [dict(m) for m in messages]
    user_idx = max(
        (i for i, m in enumerate(msgs) if m["role"] in ("user", "tool")),
        key=lambda i: len(msgs[i]["content"]),
        default=None,
    )
    if user_idx is None:
        return None
    for _ in range(12):
        try:
            if len(_template_ids(tokenizer, msgs, False)) <= max_length:
                return msgs
        except Exception:
            return None  # window the template refuses to render (e.g. no user turn)
        c = msgs[user_idx]["content"]
        if len(c) < min_keep_chars * 2:
            return None
        keep = int(len(c) * 0.35)
        msgs[user_idx]["content"] = c[:keep] + "\n[...]\n" + c[-keep:]
    return None


def windowed_examples(messages, tokenizer, max_length):
    """Split an over-long conversation into windows such that EVERY assistant turn is
    supervised exactly once and every window fits max_length. Each window keeps the
    system message (tool definitions) + a contiguous slice of turns: context grows
    backward first (max conditioning for the first supervised turn), then the window
    extends forward to supervise more turns in the same pass."""
    sys_msgs = messages[:1] if (messages and messages[0]["role"] == "system") else []
    body = messages[len(sys_msgs):]
    a_idx = [i for i, m in enumerate(body) if m["role"] == "assistant"]
    if not a_idx:
        return []

    def fits(msgs):
        try:
            return len(_template_ids(tokenizer, msgs, False)) <= max_length
        except Exception:
            return False  # e.g. Qwen template refuses windows with no user turn

    # fast path: whole conversation fits -> one window, all assistant turns supervised
    if fits(sys_msgs + body):
        out = _tokenize_with_spans(
            sys_msgs + body, tokenizer, {len(sys_msgs) + i for i in a_idx}
        )
        return [out] if out else []

    examples = []
    done = -1  # last body index already covered by a window's supervision
    while True:
        nxt = next((i for i in a_idx if i > done), None)
        if nxt is None:
            break
        # minimal valid window: the immediately-preceding turn + the assistant turn
        # (the template requires a user turn; never render an answer with no question)
        start_c = nxt - 1 if nxt > 0 else nxt
        if not fits(sys_msgs + body[start_c : nxt + 1]):
            # too big even minimal: middle-truncate the preceding user/tool turn
            shrunk = _shrink_user_middle(
                sys_msgs + body[start_c : nxt + 1], tokenizer, max_length
            )
            if shrunk is not None:
                out = _tokenize_with_spans(shrunk, tokenizer, {len(shrunk) - 1})
                if out:
                    examples.append(out)
            done = nxt
            continue
        # grow context backward from the minimal window (windows fit by construction)
        c = start_c
        while c - 1 > -1 and fits(sys_msgs + body[c - 1 : nxt + 1]):
            c -= 1
        # extend forward while it fits (supervise more turns in the same window)
        e = nxt
        while e + 1 < len(body) and fits(sys_msgs + body[c : e + 2]):
            e += 1
        sup = {len(sys_msgs) + (i - c) for i in a_idx if nxt <= i <= e}
        out = _tokenize_with_spans(sys_msgs + body[c : e + 1], tokenizer, sup)
        if out:
            examples.append(out)
        done = e
    return examples


def multiturn_sft_map_fn(batch, *, tokenizer, max_length):
    """Batched explode-map: each conversation yields >=0 windowed training rows.
    Nothing over-long is dropped wholesale — long conversations are windowed and
    oversized single turns are middle-truncated."""
    out_ids, out_labels = [], []
    for messages in batch["messages"]:
        try:
            windows = windowed_examples(messages, tokenizer, max_length)
        except Exception:
            continue  # one degenerate conversation must never kill the whole job
        for ids, labels in windows:
            out_ids.append(ids)
            out_labels.append(labels)
    return {"input_ids": out_ids, "labels": out_labels}


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
    p.add_argument("--chat_cap", type=int, default=20000,
                   help="general-chat rows per language: smoltalk (EN) + Bactrian-X id (ID); 0 disables")
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

    # --- General chat, balanced EN/ID (keeps the model conversational, not a pure
    #     tool-call reflex; ID side also reinforces Indonesian instruction following) ---
    if args.chat_cap:
        ds = datasets.load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
        ds = ds.shuffle(seed=42).select(range(min(args.chat_cap, len(ds))))
        ds = ds.map(
            lambda r: {"messages": [dict(m) for m in r["messages"]]},
            remove_columns=ds.column_names,
            num_proc=args.num_proc,
            desc="smoltalk",
        ).filter(lambda r: bool(r["messages"]), num_proc=args.num_proc)
        parts.append(ds)
        print(f"[sft-prep] smoltalk (EN chat): {len(ds):,} rows", flush=True)

        ds = datasets.load_dataset("MBZUAI/Bactrian-X", "id", split="train")
        ds = ds.shuffle(seed=42).select(range(min(args.chat_cap, len(ds))))
        ds = ds.map(
            lambda r: {"messages": bactrian_to_messages(r)},
            remove_columns=ds.column_names,
            num_proc=args.num_proc,
            desc="bactrian-id",
        ).filter(lambda r: r["messages"] is not None, num_proc=args.num_proc)
        parts.append(ds)
        print(f"[sft-prep] bactrian-x id (ID chat): {len(ds):,} rows", flush=True)

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
        batched=True,  # explode-map: long conversations window into multiple rows
        batch_size=64,
        desc="tokenize+mask+window",
    ).filter(lambda r: len(r["input_ids"]) > 0, num_proc=args.num_proc)
    print(f"[sft-prep] {len(merged):,} conversations -> {len(tokenized):,} windowed rows", flush=True)

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
