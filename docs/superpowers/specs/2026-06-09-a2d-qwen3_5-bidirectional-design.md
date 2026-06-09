# Design: `a2d-qwen3_5` — Bidirectional adapter for Qwen3.5-0.8B + MDLM training on a T4

- **Date:** 2026-06-09
- **Status:** Approved (design); pending implementation plan
- **Owner:** rifky@kolosal.ai
- **Repo:** built on top of the `dllm` clone (`ZHZisZZ/dllm`), branch `a2d-qwen3_5-bidirectional`

## 1. Goal

Adapt **`Qwen/Qwen3.5-0.8B`** into a **masked-diffusion language model** (LLaDA/MDLM-style) using the
`dllm` `a2d` (autoregressive-to-diffusion) recipe, and pre-train/fine-tune it on a **free Google Colab
T4 (16 GB, Turing sm_75)**.

The user explicitly chose Qwen3.5-0.8B despite it being unsupported by the stock `a2d` converter and
architecturally hostile to masked diffusion (see §3). This spec therefore covers building a **new custom
adapter**, not running the stock recipe.

## 2. Background: how `dllm` a2d + MDLM works (verified at code level)

The a2d MDLM pipeline has four stages:

1. **Convert** (`dllm/pipelines/a2d/convert.py`): re-instantiates a pretrained AR model under a custom
   `a2d-*` config whose model class overrides `Model.forward` to replace the causal attention mask with a
   **bidirectional, padding-only** 4D mask (`_prepare_4d_attention_mask`), then copies AR weights verbatim
   (`load_state_dict(strict=False)`). It maps `model_type → A2D config` via
   `A2D_CONFIG_MAP = {"llama", "qwen2", "qwen3"}` (convert.py:8-12,48-49). **No mask token is added and no
   embedding resize happens here.**
2. **Train** (`examples/a2d/mdlm/pt.py`, `sft.py`) with `dllm.core.trainers.MDLMTrainer`.
   The MDLM objective (`dllm/core/trainers/mdlm.py:118-212`):
   - Sample one timestep per sequence `t = ε + (1-ε)·U(0,1)`, `ε=1e-3` → with the default
     `LinearAlphaScheduler` (`α(t)=1-t`), `p_mask = t`.
   - Bernoulli-mask each maskable token (`labels != -100`) to `mask_token_id` (absorbing state).
   - Forward pass over the **noised** sequence with only the padding mask (no causal mask).
   - Per-token CE against the **clean** `input_ids`, weighted by `w(t) = -α'(t)/(1-α(t)) ≈ 1/t` and by
     `masked_mask` (only positions masked this step contribute). Normalized per maskable token.
   - **Bidirectionality is a property of the model, not the trainer** — the trainer never builds a causal
     mask. Models load via `AutoModelForMaskedLM`.
3. **Sample** (`dllm/core/samplers/mdlm.py`): LLaDA-style iterative unmasking over `steps` diffusion
   iterations, split into semi-autoregressive blocks of `block_size`, revealing top-k positions per step
   by confidence (`low_confidence` default), greedy argmax (`temperature=0`). Full bidirectional forward
   pass each step (no incremental KV cache needed).
4. **Eval** (`dllm/pipelines/a2d/eval.py`) via lm-eval harness.

The mask token (`<|mask|>`) and `eot` (`<|im_end|>` for Qwen) are added at **load time** by
`dllm.utils.get_tokenizer` (`dllm/utils/models.py`), mapping onto a pre-existing reserved vocab id — which
is why no embedding resize is needed (must be verified for the Qwen3.5 vocab; §7).

The existing reference adapter to mirror is `dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py`
(`A2DQwen3Config(model_type="a2d-qwen3")`, `A2DQwen3Model.forward` swaps the mask, `A2DQwen3LMHeadModel`,
and `AutoConfig/AutoModel/AutoModelForMaskedLM.register(...)`).

## 3. The challenge: Qwen3.5-0.8B architecture

Verified from `config.json` and `transformers/models/qwen3_5/{modeling,configuration}_qwen3_5.py`
(downloaded to `_ref/qwen3_5/`):

- **Multimodal wrapper:** checkpoint is `Qwen3_5ForConditionalGeneration` (image-text-to-text). Weights:
  `model.language_model.*` (text), `model.visual.*` (vision tower), `mtp.*` (multi-token-prediction head).
- **Text backbone** (`Qwen3_5TextConfig`, `model_type="qwen3_5_text"`): `hidden_size=1024`,
  `num_hidden_layers=24`, `head_dim=256`, `num_attention_heads=8`, `num_key_value_heads=2` (GQA),
  `intermediate_size=3584`, `vocab_size=248320`, `tie_word_embeddings=true`, RoPE (`rope_theta=1e7`,
  `partial_rotary_factor=0.25`, mRoPE), native dtype `bfloat16`.
- **Hybrid layers** (`layer_types`, `full_attention_interval=4`): **18 `linear_attention` + 6
  `full_attention`** (full at layers 3,7,11,15,19,23 — every 4th).
  - `full_attention` = `Qwen3_5Attention` (softmax, GQA, RoPE, `is_causal=True`, with an output gate
    `attn_output * sigmoid(gate)` from the `q_proj` that emits `head_dim*2`).
  - `linear_attention` = `Qwen3_5GatedDeltaNet` (gated delta-rule / DeltaNet, à la Qwen3-Next):
    a **causal depthwise conv1d** (`padding=kernel-1` then crop, `kernel=4`) followed by a **left→right
    state-recurrence scan** (`torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule`).
    `linear_num_value_heads=16`, `linear_num_key_heads=16`, key/value head dim 128.
- **Mask construction** (`Qwen3_5TextModel.forward:1188-1201`): builds `causal_mask` (→ full-attn layers)
  and `linear_attn_mask` (2D padding mask → linear-attn layers, used only by
  `apply_mask_to_padding_states`). **Directionality of the linear layers is in the conv + scan, not the
  mask.**

**Why stock a2d fails:** (1) `A2D_CONFIG_MAP["qwen3_5"]` → `KeyError`; (2) `AutoModelForCausalLM` cannot
cleanly load a `Qwen3_5ForConditionalGeneration` checkpoint; (3) swapping the mask only bidirectionalizes
the 6 full-attn layers — the 18 linear-attn layers remain strictly causal because their recurrence has no
mask to flip.

**T4 frictions:** no native bf16 (must use fp16); no flash-attn-2 (sm_75); FLA/causal-conv1d CUDA kernels
unavailable → pure-torch delta-rule fallback (slow but correct); `dllm` pins `transformers==4.57.0` but
`qwen3_5` may require a newer transformers (§7).

## 4. Locked decisions

1. **Source checkpoint:** `Qwen/Qwen3.5-0.8B-Base` (pretrained, better for continued-pretraining/MDLM)
   rather than the instruct variant.
2. **Fit strategy:** **QLoRA** (4-bit frozen base + LoRA on `all-linear`). Full FT of 0.8B + Adam (~13 GB)
   is too tight on 16 GB; QLoRA fits, and because the bidirectional mixer shares weights (Approach A),
   LoRA on the shared projections is how the model learns bidirectional behavior.
3. **Precision:** fp16 (edit dllm's hard-coded bf16). Attention impl `sdpa`. Delta-rule internals stay fp32.
4. **Staging:** gate at **Milestone 1** — prove the cheap (full-attn-only) version learns before building
   the bidirectional delta-net.
5. **Bidirectional linear attention:** **Approach A** (shared fwd+bwd sum + non-causal conv) for v1;
   escalate to Approach B (Hydra-style diagonal correction) only if quality lags.

## 5. Detailed design (staged)

### Milestone 0 — Text backbone extraction (plumbing)
- New tool that loads `Qwen/Qwen3.5-0.8B-Base`, drops `model.visual.*` and `mtp.*`, remaps
  `model.language_model.*` → text-only `Qwen3_5ForCausalLM` (`model.*` + `lm_head` / tied embeddings),
  saves model + tokenizer to `.models/qwen3_5-0.8b-text`.
- **Exit check:** the saved text-only model loads and generates coherent text causally.

### Milestone 1 — "Cheap" a2d (full-attn-only bidirectional)
- New package `dllm/pipelines/a2d/models/qwen3_5/` mirroring `qwen3/`:
  - `A2DQwen3_5TextConfig(Qwen3_5TextConfig)` with `model_type="a2d-qwen3_5"`.
  - `A2DQwen3_5TextModel(Qwen3_5TextModel)` overriding `forward` so the **full-attention** mask is the
    bidirectional padding-only mask (`_prepare_4d_attention_mask`) instead of `create_causal_mask`. The
    `linear_attn_mask` path is unchanged (linear layers stay causal in M1).
  - `A2DQwen3_5ForCausalLM(Qwen3_5ForCausalLM)` wiring the model + lm_head; `register` for `AutoConfig`,
    `AutoModel`, `AutoModelForMaskedLM`.
- Add `"qwen3_5_text": A2DQwen3_5TextConfig` to `A2D_CONFIG_MAP`; convert step consumes the M0 text model.
- `get_tokenizer`: add `<|mask|>` mask token + `<|im_end|>` eot for the new model class.
- Train MDLM (`pt.py`) on `Trelis/tiny-shakespeare`, fp16/QLoRA/sdpa.
- **Exit check / GATE:** MDLM eval NLL/PPL drops meaningfully and `sample.py` produces non-degenerate
  denoised text. If yes → proceed to M2. If no → re-evaluate the whole approach.

### Milestone 2 — Full a2d (bidirectional GatedDeltaNet)
Re-implement the linear token mixer bidirectionally (`A2DQwen3_5GatedDeltaNet`):
- **Non-causal conv:** replace `padding=kernel-1`+crop with a centered conv (`padding=kernel//2`, crop to
  seq_len) so the depthwise conv sees both neighbors.
- **Bidirectional scan (Approach A):** `out = scan(x) + flip(scan(flip(x)))` using the **same** projections
  (q,k,v,β,α,conv) for both directions; sum the outputs. Zero new parameters → LoRA-adaptable.
- **Remove the cache/incremental-decode path** (`use_precomputed_states`, `conv_state`, `recurrent_state`)
  — masked diffusion always does full forward passes; no causal KV/conv cache.
- **Padding:** keep `apply_mask_to_padding_states`; for right-padded SFT batches, restrict each direction's
  scan to valid token positions (PT uses packed fixed-length sequences with no padding, so M2 can land on
  PT first and handle SFT padding second).
- Retrain; compare M1 vs M2 (does bidirectionalizing the 18 linear layers improve NLL/sample quality?).
- **Fallback:** if Approach A underfits, escalate to Approach B (subtract the double-counted diagonal /
  self-term for a cleaner quasiseparable bidirectional mixer).

### dllm integration points (file-level)
- `dllm/pipelines/a2d/models/qwen3_5/{__init__,configuration,modeling}.py` (new).
- `dllm/pipelines/a2d/__init__.py` — export new classes.
- `dllm/pipelines/a2d/convert.py` — add map entry; support a `--text_extract` path (or rely on M0 tool).
- `dllm/utils/models.py` (`get_tokenizer`) — mask/eot tokens for the new class.
- `dllm/utils/configs.py:14,63` and `dllm/utils/models.py:32,58` — fp16 instead of bf16 (or expose flags
  and pass `--dtype float16 --bf16 False --fp16 True`).
- `scripts/tests/test_attention.py` — add an `a2d` bidirectional-attention test for qwen3_5
  (`pytest -k "test_a2d"` analogue) that asserts a change at position `i` affects outputs at `j < i`.

## 6. Training config (free Colab T4, 16 GB)
- QLoRA: `--load_in_4bit True --lora True` (`r=32, alpha=64, dropout=0.05, target_modules=all-linear`,
  `bnb_4bit_compute_dtype=float16`). Requires `bitsandbytes` (optional extra).
- `--bf16 False --fp16 True --dtype float16 --attn_implementation sdpa`.
- `--per_device_train_batch_size 1 --gradient_accumulation_steps 8..16 --max_length 128`
  (packed `tokenize_and_group`), `--gradient_checkpointing True`.
- `--learning_rate 1e-4` cosine, `warmup_ratio 0.1`, `--report_to none`.
- Data: `Trelis/tiny-shakespeare` (smoke) → streaming `dylanebert/openwebtext` once milestones pass.
- **Colab survival:** checkpoint to Google Drive on a step interval; resume from `checkpoint-final` across
  ~12 h disconnects. Edit `download_hf_*` tools' hardcoded Lustre paths to use HF cache / Drive.

## 7. Environment & setup
- Colab T4, Python 3.10/3.11, torch with CUDA (cu121/cu124), **no flash-attn**.
- **transformers:** needs a version that ships `qwen3_5` (the config reports `4.57.0.dev0`; it is on
  transformers `main`). `dllm` pins `transformers==4.57.0` — **must verify** whether 4.57.0 includes
  `qwen3_5`; if not, bump transformers (e.g. to a 4.57.x / main build that has it) and confirm `dllm`
  still imports/trains. This is a tracked risk.
- `bitsandbytes` for 4-bit; `peft` for LoRA (already a dllm dep).

## 8. Validation
- **M0:** causal greedy generation is coherent.
- **M1/M2:** MDLM eval NLL/PPL trend down; `sample.py`/`chat.py` (low-confidence remasking, ~128 steps)
  produce increasingly coherent denoised text; quantitative M1-vs-M2 comparison on held-out NLL.
- **Bidirectionality unit test:** perturbing a future token changes a past token's logits (full model);
  for M2, the same test passes through a single GatedDeltaNet layer in isolation.

## 9. Risks & open questions
- **Convergence (highest risk):** bidirectionalizing a pretrained causal delta-net with shared weights is
  novel; it may not adapt well via LoRA alone. Mitigations: gate at M1; Approach B fallback; consider
  un-freezing the delta-net norm/conv params even under QLoRA.
- **transformers version vs dllm pin** (§7).
- **4-bit + custom modeling + fp32 scan** interplay (bitsandbytes quantizes Linear layers; the delta-rule
  scan casts to fp32 internally — verify numerics and that `out_proj`/`in_proj_*` quantize cleanly).
- **Pure-torch delta-rule speed on T4** (Python chunk loop) — may be slow; acceptable for small
  seq/batch, revisit if throughput is unworkable.
- **Mask token id availability:** confirm the 248320 Qwen3.5 vocab has a reserved/free id for `<|mask|>`
  and `<|im_end|>` so no embedding resize is needed.
- **SFT padding** in the bidirectional scan (handled after PT lands).
- **Approach A double-counts** the current token's self-contribution (fwd+bwd) — acceptable for v1, fixed
  in Approach B.

## 10. Out of scope
Vision/multimodal, MoE (`qwen3_5_moe`), multi-GPU/DeepSpeed/FSDP, RL/GRPO, BD3LM (block diffusion), exact
reproduction of any released Tiny-A2D model.
