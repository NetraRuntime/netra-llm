"""
Held-out per-domain BD3LM NELBO eval.

Streams UNSEEN documents from each PT source (offsets past the training slices), packs
to the training seq length, and computes the exact training-loss estimator (random-t
masking, noised|clean concat, block-causal mask, scheduler-weighted CE over masked
positions) with a FIXED seed — comparable across checkpoints of the same arch.

Usage:
    python dllm/tools/eval_bd3lm_nll.py --model_path <ckpt> [--rows 64] [--draws 4]
"""

import argparse
import functools

import torch
import datasets
import dllm
from dllm.core.trainers.bd3lm import _create_bd3lm_attention_mask
from dllm.core.schedulers import LinearAlphaScheduler

# (name, repo, config, docs to SKIP = the training slice, docs to take)
DOMAINS = [
    ("english", "HuggingFaceFW/fineweb-edu", "sample-10BT", 240_000, 3000),
    ("indonesian", "HuggingFaceFW/fineweb-2", "ind_Latn", 620_000, 3000),
    ("code", "OpenCoder-LLM/opc-fineweb-code-corpus", None, 420_000, 3000),
]


@torch.no_grad()
def domain_nll(model, tok, texts, seq_len, rows, draws, block_size, device):
    packed = dllm.utils.tokenize_and_group(
        {"text": texts}, tokenizer=tok, text_field="text", seq_length=seq_len,
        insert_eos=True, drop_tail=True,
    )["input_ids"][:rows]
    sched = LinearAlphaScheduler()
    l = seq_len
    idx = torch.arange(2 * l, device=device)
    attn = _create_bd3lm_attention_mask(
        None, None, idx[:, None], idx[None, :], block_size=block_size, n=l
    ).unsqueeze(0).unsqueeze(0)
    pos = torch.cat([torch.arange(l), torch.arange(l)]).unsqueeze(0).to(device)
    g = torch.Generator(device="cpu").manual_seed(1234)

    tot, cnt = 0.0, 0.0
    for r in range(0, len(packed), 4):
        batch = torch.tensor(packed[r : r + 4], device=device)
        b = batch.shape[0]
        for _ in range(draws):
            t = 1e-3 + (1 - 1e-3) * torch.rand(b, generator=g).to(device)
            p_mask = (1.0 - sched._alpha(t)).unsqueeze(1).expand(b, l)
            masked = torch.rand(b, l, generator=g).to(device) < p_mask
            noised = torch.where(masked, tok.mask_token_id, batch)
            concat = torch.cat([noised, batch], dim=1)
            out = model(
                input_ids=concat,
                attention_mask=attn,
                position_ids=pos.expand(b, -1),
                logits_to_keep=slice(0, l),
            )
            w = (-sched._alpha_derivative(t) / (1 - sched._alpha(t) + 1e-6)).unsqueeze(1)
            ce = torch.nn.functional.cross_entropy(
                out.logits.reshape(-1, out.logits.size(-1)).float(),
                batch.reshape(-1),
                reduction="none",
            ).view(b, l)
            ce = ce * w.expand(b, l) * masked.float()
            tot += ce.sum().item()
            cnt += masked.sum().item()
    return tot / max(cnt, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--block_size", type=int, default=32)
    args = p.parse_args()

    device = "cuda"
    model = dllm.utils.get_model(
        model_name_or_path=args.model_path, dtype="bfloat16", attn_implementation="sdpa"
    ).to(device).eval()
    tok = dllm.utils.get_tokenizer(model_name_or_path=args.model_path)

    for name, repo, cfg, skip, take in DOMAINS:
        ds = datasets.load_dataset(repo, name=cfg, split="train", streaming=True)
        texts = [ex["text"] for ex in ds.skip(skip).take(take)]
        nll = domain_nll(model, tok, texts, args.seq_len, args.rows, args.draws,
                         args.block_size, device)
        print(f"[nll] {args.model_path} {name}: {nll:.4f}", flush=True)


if __name__ == "__main__":
    main()
