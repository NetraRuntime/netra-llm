# Colab T4 Runbook — `a2d-qwen3_5` masked-diffusion training

Runs the full pipeline for adapting **Qwen3.5-0.8B-Base** into a masked-diffusion (MDLM)
model on a free Colab **T4 (16 GB)**. See the design spec and plan under
`docs/superpowers/`.

**Prereq:** this branch (`a2d-qwen3_5-bidirectional`) must live on a remote you can clone
on Colab (push it to your GitHub fork first).

> Memory note: full FT of 0.8B won't fit a T4 — this uses **QLoRA (4-bit) + fp16 + sdpa**,
> batch 1 × grad-accum 8, `max_length 128`, gradient checkpointing. If you OOM, drop
> `--max_length` to 64.

---

## Cell 1 — clone, env, Drive
```bash
!git clone -b a2d-qwen3_5-bidirectional https://github.com/<YOUR_FORK>/dllm.git
%cd dllm
!bash scripts/colab/setup.sh          # installs dllm + transformers(main, has qwen3_5) + bitsandbytes
```
```python
from google.colab import drive; drive.mount('/content/drive')
import os
os.environ["BASE"] = "/content/drive/MyDrive/a2d-qwen3_5"
os.makedirs(os.environ["BASE"], exist_ok=True)
```

## Cell 2 — M0: extract the text backbone (once; cached to Drive)
```bash
!python dllm/tools/extract_qwen3_5_text.py \
    --model_name_or_path "Qwen/Qwen3.5-0.8B-Base" \
    --output_dir "$BASE/qwen3_5-0.8b-text"
```
Sanity-check causal generation (should be grammatical):
```bash
!python - <<'PY'
import transformers, torch, os
p=os.environ["BASE"]+"/qwen3_5-0.8b-text"
m=transformers.AutoModelForCausalLM.from_pretrained(p, dtype=torch.float16).cuda().eval()
t=transformers.AutoTokenizer.from_pretrained(p)
ids=t("The capital of France is", return_tensors="pt").input_ids.cuda()
print(t.decode(m.generate(ids, max_new_tokens=20)[0]))
PY
```

## Cell 3 — M1: convert to the a2d diffusion model (linear layers stay causal)
```bash
!python dllm/pipelines/a2d/convert.py \
    --model_name_or_path "$BASE/qwen3_5-0.8b-text" \
    --output_dir "$BASE/a2d/qwen3_5-0.8b"
```

## Cell 4 — M1: train (resumable; re-run this exact cell after a disconnect)
```bash
!bash scripts/colab/train_qwen3_5_mdlm.sh "$BASE/a2d/qwen3_5-0.8b" "$BASE/runs/m1-tinyshake" 10
```

## Cell 5 — M1 GATE: sample and eyeball
```bash
!python -u examples/a2d/mdlm/sample.py \
    --model_name_or_path "$BASE/runs/m1-tinyshake/checkpoint-final" \
    --dtype float16 --attn_implementation sdpa \
    --steps 128 --max_new_tokens 128 --remasking low_confidence --temperature 0.0
```
**Decision:** if the loss dropped and samples are increasingly Shakespeare-like → proceed to
M2. If flat/degenerate after a real run → stop and reassess (the full-attn-only signal may be
too weak).

## Cell 6 — M2: enable bidirectional linear attention, convert, train, compare
```bash
# Convert into a fresh dir, then flip the config flag on.
!python dllm/pipelines/a2d/convert.py \
    --model_name_or_path "$BASE/qwen3_5-0.8b-text" \
    --output_dir "$BASE/a2d/qwen3_5-0.8b-bidir"
```
```python
import json, os
cfg_path = os.environ["BASE"] + "/a2d/qwen3_5-0.8b-bidir/config.json"
cfg = json.load(open(cfg_path)); cfg["bidirectional_linear"] = True
json.dump(cfg, open(cfg_path, "w"), indent=2)
print("bidirectional_linear =", cfg["bidirectional_linear"])
```
```bash
!bash scripts/colab/train_qwen3_5_mdlm.sh "$BASE/a2d/qwen3_5-0.8b-bidir" "$BASE/runs/m2-tinyshake" 10
```
Then compare M1 vs M2 final eval NLL/PPL and sample quality (Cell 5 against the
`m2-tinyshake/checkpoint-final`). If M2 ≤ M1 or unstable, see the spec's Approach-B
(diagonal-correction) fallback.

---

### Notes / known limitations
- The `bitsandbytes` 4-bit path requires a CUDA GPU (T4 is fine); it is not used in the
  CPU unit tests.
- The gated delta-rule runs the **pure-torch** kernels (no `fla`/`causal-conv1d` on T4) — correct
  but slower than fused kernels.
- **SFT/padding:** the bidirectional backward scan is only correct for *unpadded* (packed PT)
  sequences. Right-padded SFT batches contaminate the backward scan — SFT is deferred until a
  masked-reverse-scan is added (see spec §9).
- `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask` is deprecated in very new
  transformers; if a future Colab transformers removes it, migrate the mask build in
  `A2DQwen3_5TextModel.forward` to `transformers.masking_utils` (matches the sibling `qwen3` adapter).
