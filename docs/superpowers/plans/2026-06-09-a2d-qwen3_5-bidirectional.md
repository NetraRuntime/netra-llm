# a2d-qwen3_5 Bidirectional Adapter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt `Qwen/Qwen3.5-0.8B-Base` into a masked-diffusion (MDLM) language model via `dllm`'s a2d recipe and train it on a free Colab T4, building the custom bidirectional adapter the stock recipe lacks.

**Architecture:** Extract the text backbone from the multimodal checkpoint into a `Qwen3_5ForCausalLM` (M0); subclass it into `A2DQwen3_5` that makes the 6 full-attention layers bidirectional via a padding-only mask while leaving the 18 GatedDeltaNet layers causal (M1, the gate); then re-implement the GatedDeltaNet token mixer bidirectionally (forward+backward scan, non-causal conv) (M2). Train with QLoRA + fp16 + sdpa.

**Tech Stack:** PyTorch, `transformers` (with `qwen3_5`), `dllm`, `peft` (LoRA), `bitsandbytes` (4-bit), `accelerate`, `datasets`. Hardware: single 16 GB T4 (Turing sm_75).

**Spec:** `docs/superpowers/specs/2026-06-09-a2d-qwen3_5-bidirectional-design.md`

**Naming contract (used across all tasks):**
- New package: `dllm/pipelines/a2d/models/qwen3_5/modeling_qwen3_5.py`
- Classes: `A2DQwen3_5TextConfig` (`model_type = "a2d-qwen3_5"`), `A2DQwen3_5TextModel`, `A2DQwen3_5LMHeadModel`, `A2DQwen3_5GatedDeltaNet` (M2 only)
- Source text `model_type` after extraction = `"qwen3_5_text"` → this is the `A2D_CONFIG_MAP` key.
- Mask token: `"<|mask|>"` (added + embeddings resized during extraction). EOT: `"<|im_end|>"` (id 248046).
- Extraction tool: `dllm/tools/extract_qwen3_5_text.py`
- Paths: text ckpt `.models/qwen3_5-0.8b-text`; a2d ckpt `.models/a2d/qwen3_5-0.8b`; trained `.models/a2d/qwen3_5-0.8b/mdlm/tiny-shakespeare`.

---

## Phase A — Environment (critical path: transformers must ship `qwen3_5` AND import `dllm`)

### Task A1: Pin a transformers that has `qwen3_5` and verify dllm still imports

**Files:**
- Modify: `pyproject.toml:15` (relax `transformers==4.57.0` pin)
- Create: `scripts/colab/setup.sh`

- [ ] **Step 1: Write the failing verification test**

Create `scripts/tests/test_env_qwen3_5.py`:

```python
def test_transformers_has_qwen3_5():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: F401

def test_dllm_imports_with_this_transformers():
    import dllm  # noqa: F401
    import dllm.pipelines.a2d  # noqa: F401
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest scripts/tests/test_env_qwen3_5.py -v`
Expected: FAIL on `test_transformers_has_qwen3_5` with `ImportError: cannot import name 'Qwen3_5ForCausalLM'` (stock dllm pins transformers 4.57.0 which predates `qwen3_5`).

- [ ] **Step 3: Relax the pin and install a transformers with `qwen3_5`**

Edit `pyproject.toml` line 15: change `"transformers==4.57.0",` to `"transformers>=4.57.0",`.

Create `scripts/colab/setup.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
# dllm deps first
pip install -e .
# transformers build that ships qwen3_5 (overrides the relaxed pin)
pip install --upgrade "git+https://github.com/huggingface/transformers.git"
# QLoRA deps (optional extra in dllm)
pip install bitsandbytes==0.48.1
# T4: NO flash-attn (sm_75 unsupported). sdpa is used instead.
python - <<'PY'
import transformers, torch
from transformers import Qwen3_5ForCausalLM
print("transformers", transformers.__version__, "torch", torch.__version__)
print("qwen3_5 OK")
PY
```

- [ ] **Step 4: Run setup and the test to verify both pass**

Run: `bash scripts/colab/setup.sh && pytest scripts/tests/test_env_qwen3_5.py -v`
Expected: PASS (both tests). If `import dllm` breaks under newer transformers, record the exact error — a stock-dllm-vs-new-transformers incompatibility is a gating risk and must be fixed before continuing (patch the offending dllm import).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml scripts/colab/setup.sh scripts/tests/test_env_qwen3_5.py
git commit -m "build: allow transformers>=4.57.0 (qwen3_5) and add env smoke test"
```

---

## Phase B — Milestone 0: extract the text backbone

### Task B2: Write the text-extraction tool

**Files:**
- Create: `dllm/tools/extract_qwen3_5_text.py`
- Test: `scripts/tests/test_extract_qwen3_5_text.py`

The multimodal checkpoint stores text weights under `model.language_model.*`, vision under `model.visual.*`, plus `mtp.*`. We load the multimodal model, copy `model.language_model` into a text-only `Qwen3_5ForCausalLM.model`, tie the head, add `<|mask|>`, resize embeddings, and save.

- [ ] **Step 1: Write the failing test**

```python
# scripts/tests/test_extract_qwen3_5_text.py
import torch, transformers
from dllm.tools.extract_qwen3_5_text import extract_text_backbone

def test_extract_produces_causal_lm_with_mask_token(tmp_path):
    # Build a tiny multimodal model from the real config shrunk down, save, extract.
    cfg = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
    cfg.text_config.num_hidden_layers = 4         # 3 linear + 1 full (interval 4)
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest scripts/tests/test_extract_qwen3_5_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dllm.tools.extract_qwen3_5_text'`.

- [ ] **Step 3: Implement the tool**

```python
# dllm/tools/extract_qwen3_5_text.py
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest scripts/tests/test_extract_qwen3_5_text.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dllm/tools/extract_qwen3_5_text.py scripts/tests/test_extract_qwen3_5_text.py
git commit -m "feat(a2d): extract Qwen3.5 text backbone into Qwen3_5ForCausalLM + mask token"
```

### Task B3: Run real extraction + causal sanity check (manual, GPU/Colab)

- [ ] **Step 1: Extract the real 0.8B text backbone**

Run: `python dllm/tools/extract_qwen3_5_text.py --model_name_or_path "Qwen/Qwen3.5-0.8B-Base" --output_dir ".models/qwen3_5-0.8b-text"`
Expected: saves `config.json` (`model_type: qwen3_5_text`, `vocab_size: 248321`), safetensors, tokenizer. `missing`/`unexpected` lists are empty for `text.model`.

- [ ] **Step 2: Verify coherent causal generation**

Run:
```bash
python - <<'PY'
import transformers, torch
m = transformers.AutoModelForCausalLM.from_pretrained(".models/qwen3_5-0.8b-text", dtype=torch.float16).cuda().eval()
t = transformers.AutoTokenizer.from_pretrained(".models/qwen3_5-0.8b-text")
ids = t("The capital of France is", return_tensors="pt").input_ids.cuda()
print(t.decode(m.generate(ids, max_new_tokens=20)[0]))
PY
```
Expected: grammatical continuation (e.g. mentions "Paris"). If gibberish, the weight remap is wrong — STOP and fix before M1.

---

## Phase C — Milestone 1: full-attention-only bidirectional adapter (THE GATE)

### Task C1: Create the `A2DQwen3_5` adapter (full-attn bidirectional, linear stays causal)

**Files:**
- Create: `dllm/pipelines/a2d/models/qwen3_5/modeling_qwen3_5.py`

This mirrors `Qwen3_5TextModel.forward` (transformers) but replaces `create_causal_mask(...)` with a bidirectional padding-only mask fed to the full-attention layers. The `linear_attn_mask` path is untouched (GatedDeltaNet stays causal in M1).

- [ ] **Step 1: Write the modeling file**

```python
# dllm/pipelines/a2d/models/qwen3_5/modeling_qwen3_5.py
from typing import Optional

import torch
from torch import nn

import transformers
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5TextModel,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


class A2DQwen3_5TextConfig(Qwen3_5TextConfig):
    model_type = "a2d-qwen3_5"  # <- NEW model_type
    # M2 toggle (ignored in M1): when True, GatedDeltaNet runs bidirectionally.
    def __init__(self, bidirectional_linear: bool = False, **kwargs):
        self.bidirectional_linear = bidirectional_linear
        super().__init__(**kwargs)


class A2DQwen3_5TextModel(Qwen3_5TextModel):
    config_class = A2DQwen3_5TextConfig

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Qwen3_5ModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # mRoPE position ids (text only): replicate the stock 4-way expand.
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        # ---- NEW CODE: bidirectional, padding-only mask for FULL-ATTENTION layers ----
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long
            )
        full_attn_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
        # linear-attention layers: keep stock 2D padding-mask handling (causal in M1)
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)
        # -----------------------------------------------------------------------------

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = (
                linear_attn_mask
                if self.config.layer_types[i] == "linear_attention"
                else full_attn_mask
            )
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=None,        # diffusion: no incremental cache
                use_cache=False,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return Qwen3_5ModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None)


class A2DQwen3_5LMHeadModel(Qwen3_5ForCausalLM):
    config_class = A2DQwen3_5TextConfig
    config: A2DQwen3_5TextConfig

    def __init__(self, config):
        transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = A2DQwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # belt-and-suspenders: ensure full-attn modules never self-select causal
        for module in self.modules():
            if hasattr(module, "is_causal"):
                module.is_causal = False
        self.post_init()


transformers.AutoConfig.register("a2d-qwen3_5", A2DQwen3_5TextConfig)
transformers.AutoModel.register(A2DQwen3_5TextConfig, A2DQwen3_5LMHeadModel)
transformers.AutoModelForMaskedLM.register(A2DQwen3_5TextConfig, A2DQwen3_5LMHeadModel)
```

- [ ] **Step 2: Commit (no test yet — wired up in C2/C3)**

```bash
git add dllm/pipelines/a2d/models/qwen3_5/modeling_qwen3_5.py
git commit -m "feat(a2d): A2DQwen3_5 adapter (full-attn bidirectional, linear causal)"
```

### Task C2: Wire the adapter into dllm (exports, convert map, tokenizer)

**Files:**
- Modify: `dllm/pipelines/a2d/__init__.py`
- Modify: `dllm/pipelines/a2d/convert.py:8-12`
- Modify: `dllm/utils/models.py:102-106,207`

- [ ] **Step 1: Export the new classes**

In `dllm/pipelines/a2d/__init__.py`, add this import line:

```python
from .models.qwen3_5.modeling_qwen3_5 import A2DQwen3_5TextConfig, A2DQwen3_5LMHeadModel
```

and append `"A2DQwen3_5TextConfig"`, `"A2DQwen3_5LMHeadModel"` to `__all__`.

- [ ] **Step 2: Add to the converter map**

In `dllm/pipelines/a2d/convert.py`, update `A2D_CONFIG_MAP` (lines 8-12):

```python
A2D_CONFIG_MAP = {
    "llama": dllm.pipelines.a2d.A2DLlamaConfig,
    "qwen2": dllm.pipelines.a2d.A2DQwen2Config,
    "qwen3": dllm.pipelines.a2d.A2DQwen3Config,
    "qwen3_5_text": dllm.pipelines.a2d.A2DQwen3_5TextConfig,
}
```

- [ ] **Step 3: Add tokenizer customization**

In `dllm/utils/models.py`, extend the import (lines 102-106) and the A2D-Qwen branch (line 207):

```python
    from dllm.pipelines.a2d import (
        A2DLlamaLMHeadModel,
        A2DQwen2LMHeadModel,
        A2DQwen3LMHeadModel,
        A2DQwen3_5LMHeadModel,
    )
```

and change the branch condition at line 207 to include the new class:

```python
    elif issubclass(model_cls, (A2DQwen2LMHeadModel, A2DQwen3LMHeadModel, A2DQwen3_5LMHeadModel)):
```

(The body — `add_special_tokens({"mask_token": "<|mask|>"})`, `eot_token = "<|im_end|>"`, enable_thinking wrapper — is correct as-is; since extraction already added `<|mask|>`, `add_special_tokens` is a no-op that just sets `mask_token`.)

- [ ] **Step 4: Verify imports resolve**

Run: `python -c "import dllm; print(dllm.pipelines.a2d.A2DQwen3_5LMHeadModel, dllm.pipelines.a2d.A2DQwen3_5TextConfig)"`
Expected: prints both classes, no ImportError.

- [ ] **Step 5: Commit**

```bash
git add dllm/pipelines/a2d/__init__.py dllm/pipelines/a2d/convert.py dllm/utils/models.py
git commit -m "feat(a2d): wire A2DQwen3_5 into exports, convert map, tokenizer"
```

### Task C3: Bidirectionality test (`future_affects_past`) for qwen3_5 — THE GATE CRITERION

**Files:**
- Create: `scripts/tests/attention/test_invariance_qwen3_5.py`

In M1 only the 6 full-attn layers are bidirectional, so the cross-token padding-invariance test (`_assert_invariance`) is **not** expected to hold for the causal linear layers — we therefore test only the directional property: perturbing a future token must change a past token's logits (which it does via the full-attn layers). We use a tiny random model so it runs on CPU/T4 fast.

- [ ] **Step 1: Write the failing test**

```python
# scripts/tests/attention/test_invariance_qwen3_5.py
import torch
import transformers
import dllm
from .common import ERROR_THRESHOLD


def _tiny_a2d_qwen3_5():
    base = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B-Base").text_config
    cfg = dllm.pipelines.a2d.A2DQwen3_5TextConfig(
        **{**base.to_dict(), "num_hidden_layers": 4, "hidden_size": 128,
           "head_dim": 32, "intermediate_size": 256}
    )
    torch.manual_seed(0)
    return dllm.pipelines.a2d.A2DQwen3_5LMHeadModel(cfg).eval()


def test_a2d_qwen3_5_future_affects_past():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _tiny_a2d_qwen3_5().to(device)
    a = torch.tensor([[101, 102, 103, 104]], device=device)
    b = torch.tensor([[101, 102, 999, 104]], device=device)  # perturb position 2 (future of 1)
    with torch.no_grad():
        la = model(a).logits
        lb = model(b).logits
    diff = (la[:, 1, :] - lb[:, 1, :]).abs().max().item()
    assert diff > ERROR_THRESHOLD, f"not bidirectional, diff={diff}"
```

- [ ] **Step 2: Run it to verify it fails (before C1/C2 are in place) or passes**

Run: `pytest scripts/tests/attention/test_invariance_qwen3_5.py -v`
Expected: PASS once C1+C2 are committed. If it FAILS (`diff` ≈ 0), the full-attn mask swap is not active — debug `A2DQwen3_5TextModel.forward` (is `full_attn_mask` reaching the full-attn layers? is `is_causal` still True?).

- [ ] **Step 3: Commit**

```bash
git add scripts/tests/attention/test_invariance_qwen3_5.py
git commit -m "test(a2d): qwen3_5 future-affects-past bidirectionality gate"
```

### Task C4: Convert the extracted text model → a2d diffusion model

- [ ] **Step 1: Run the converter (manual)**

Run: `python dllm/pipelines/a2d/convert.py --model_name_or_path ".models/qwen3_5-0.8b-text" --output_dir ".models/a2d/qwen3_5-0.8b"`
Expected: prints empty/near-empty `missing`/`unexpected`; saves a checkpoint with `model_type: a2d-qwen3_5`.

- [ ] **Step 2: Verify it loads as a masked LM**

Run: `python -c "import dllm, transformers, torch; m=transformers.AutoModelForMaskedLM.from_pretrained('.models/a2d/qwen3_5-0.8b', dtype=torch.float16); print(type(m).__name__, m.config.model_type)"`
Expected: `A2DQwen3_5LMHeadModel a2d-qwen3_5`.

### Task C5: Smoke MDLM training on T4 (QLoRA + fp16 + sdpa)

- [ ] **Step 1: Launch a short training run**

Run:
```bash
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/a2d/mdlm/pt.py \
    --model_name_or_path ".models/a2d/qwen3_5-0.8b" \
    --dataset_args "Trelis/tiny-shakespeare" --text_field "Text" --insert_eos False \
    --max_length 128 \
    --dtype float16 --bf16 False --fp16 True --attn_implementation sdpa \
    --load_in_4bit True --lora True \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --max_steps 100 --learning_rate 1e-4 --eval_strategy no --report_to none \
    --output_dir ".models/a2d/qwen3_5-0.8b/mdlm/tiny-shakespeare"
```
Expected: trains 100 steps without OOM; the logged `loss` trends downward from its start value. Record peak GPU memory (`nvidia-smi`).

- [ ] **Step 2: If OOM**, reduce: `--max_length 64`, ensure `--gradient_checkpointing True`, keep batch 1. If still OOM, the 0.8B QLoRA footprint is too large for this T4 — document and escalate (e.g. CPU-offload optimizer via a zero2 config edited to fp16).

### Task C6: Sample from the M1 checkpoint — GATE DECISION

- [ ] **Step 1: Sample**

Run:
```bash
python -u examples/a2d/mdlm/sample.py \
    --model_name_or_path ".models/a2d/qwen3_5-0.8b/mdlm/tiny-shakespeare/checkpoint-final" \
    --dtype float16 --attn_implementation sdpa \
    --steps 128 --max_new_tokens 128 --remasking low_confidence --temperature 0.0
```
Expected (after only 100 smoke steps): not necessarily coherent, but **non-degenerate** (not all one token, not all mask). For a real gate decision, train longer (e.g. `--num_train_epochs 10`, no `--max_steps`) and re-sample.

- [ ] **Step 2: GATE.** If loss drops and samples become increasingly Shakespeare-like → proceed to Milestone 2. If loss is flat or samples are degenerate after a real run → STOP and reassess (the full-attn-only signal may be too weak; revisit before investing in M2).

---

## Phase D — Milestone 2: bidirectional GatedDeltaNet (gated on C6)

### Task D1: Unit test — a single GatedDeltaNet layer is bidirectional in isolation

**Files:**
- Create: `scripts/tests/attention/test_qwen3_5_gated_deltanet.py`

- [ ] **Step 1: Write the failing test**

```python
# scripts/tests/attention/test_qwen3_5_gated_deltanet.py
import torch
import transformers
import dllm
from .common import ERROR_THRESHOLD


def _layer(bidirectional):
    base = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B-Base").text_config
    cfg = dllm.pipelines.a2d.A2DQwen3_5TextConfig(
        **{**base.to_dict(), "hidden_size": 128, "head_dim": 32},
        bidirectional_linear=bidirectional,
    )
    from dllm.pipelines.a2d.models.qwen3_5.modeling_qwen3_5 import A2DQwen3_5GatedDeltaNet
    torch.manual_seed(0)
    return A2DQwen3_5GatedDeltaNet(cfg, layer_idx=0).eval()


def test_gated_deltanet_bidirectional_future_affects_past():
    torch.manual_seed(0)
    layer = _layer(bidirectional=True)
    x = torch.randn(1, 6, 128)
    x2 = x.clone(); x2[:, 4, :] += 1.0   # perturb a FUTURE position
    with torch.no_grad():
        o, o2 = layer(x), layer(x2)
    diff_past = (o[:, 1, :] - o2[:, 1, :]).abs().max().item()
    assert diff_past > ERROR_THRESHOLD, f"deltanet not bidirectional, diff={diff_past}"


def test_gated_deltanet_causal_when_disabled():
    torch.manual_seed(0)
    layer = _layer(bidirectional=False)
    x = torch.randn(1, 6, 128)
    x2 = x.clone(); x2[:, 4, :] += 1.0
    with torch.no_grad():
        o, o2 = layer(x), layer(x2)
    diff_past = (o[:, 1, :] - o2[:, 1, :]).abs().max().item()
    assert diff_past < ERROR_THRESHOLD, f"expected causal, diff={diff_past}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest scripts/tests/attention/test_qwen3_5_gated_deltanet.py -v`
Expected: FAIL — `A2DQwen3_5GatedDeltaNet` does not exist yet.

### Task D2: Implement the bidirectional GatedDeltaNet

**Files:**
- Modify: `dllm/pipelines/a2d/models/qwen3_5/modeling_qwen3_5.py`

Subclass the stock `Qwen3_5GatedDeltaNet`, drop the cache path, make the conv non-causal, and run the delta-rule scan forward + backward (shared weights), summing the outputs (Approach A).

- [ ] **Step 1: Add the class (append imports + class to the modeling file)**

Add to the imports:

```python
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    apply_mask_to_padding_states,
)
```

Add the class:

```python
class A2DQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Bidirectional gated delta-net for masked diffusion (no KV/conv cache).

    Approach A: run the causal delta-rule scan left->right and right->left with
    SHARED weights, sum the outputs; use a non-causal (centered) depthwise conv.
    """

    def _noncausal_conv(self, mixed_qkv):
        # mixed_qkv: [b, conv_dim, T]. Stock conv is causal (padding=k-1, crop left).
        T = mixed_qkv.shape[-1]
        k = self.conv_kernel_size
        out = F.conv1d(
            mixed_qkv, self.conv1d.weight, self.conv1d.bias,
            padding=k // 2, groups=self.conv_dim,
        )[..., :T]
        return F.silu(out)

    def _scan(self, query, key, value, g, beta):
        core, _ = self.chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        return core  # [b, T, heads, head_v_dim]

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        b, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)        # [b, conv_dim, T]
        z = self.in_proj_z(hidden_states).reshape(b, seq_len, -1, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()                      # [b, T, num_v]
        g = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias
        )                                                                  # [b, T, num_v]

        mixed_qkv = self._noncausal_conv(mixed_qkv).transpose(1, 2)        # [b, T, conv_dim]
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(b, seq_len, -1, self.head_k_dim)
        key = key.reshape(b, seq_len, -1, self.head_k_dim)
        value = value.reshape(b, seq_len, -1, self.head_v_dim)
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        out_fwd = self._scan(query, key, value, g, beta)
        if getattr(self.config, "bidirectional_linear", False):
            flip = lambda t: torch.flip(t, dims=[1])
            out_bwd = self._scan(flip(query), flip(key), flip(value), flip(g), flip(beta))
            core_attn_out = out_fwd + torch.flip(out_bwd, dims=[1])
        else:
            core_attn_out = out_fwd

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(b, seq_len, -1)
        return self.out_proj(core_attn_out)
```

Note: `A2DQwen3_5GatedDeltaNet` stores `self.config = config` — add `self.config = config` in a light `__init__` override if the stock base does not retain it (stock `Qwen3_5GatedDeltaNet.__init__` reads from config but does not store it). Add:

```python
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.config = config
```

- [ ] **Step 2: Run the D1 unit test to verify it passes**

Run: `pytest scripts/tests/attention/test_qwen3_5_gated_deltanet.py -v`
Expected: PASS both (`bidirectional` perturbs the past; `disabled` does not).

- [ ] **Step 3: Commit**

```bash
git add dllm/pipelines/a2d/models/qwen3_5/modeling_qwen3_5.py scripts/tests/attention/test_qwen3_5_gated_deltanet.py
git commit -m "feat(a2d): bidirectional GatedDeltaNet (Approach A) behind config flag"
```

### Task D3: Use the bidirectional layer in `A2DQwen3_5TextModel`

**Files:**
- Modify: `dllm/pipelines/a2d/models/qwen3_5/modeling_qwen3_5.py`

- [ ] **Step 1: Swap the linear-attn module construction**

In `A2DQwen3_5TextModel.__init__` (override it), after `super().__init__(config)`, replace each `linear_attention` layer's `linear_attn` with the bidirectional class when `config.bidirectional_linear`:

```python
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "bidirectional_linear", False):
            for i, layer in enumerate(self.layers):
                if config.layer_types[i] == "linear_attention":
                    layer.linear_attn = A2DQwen3_5GatedDeltaNet(config, layer_idx=i)
            self.post_init()
```

- [ ] **Step 2: Extend the C3 gate test to the bidirectional model**

Add to `scripts/tests/attention/test_invariance_qwen3_5.py`:

```python
def test_a2d_qwen3_5_bidirectional_linear_future_affects_past():
    import transformers, torch, dllm
    base = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B-Base").text_config
    cfg = dllm.pipelines.a2d.A2DQwen3_5TextConfig(
        **{**base.to_dict(), "num_hidden_layers": 3, "hidden_size": 128,
           "head_dim": 32, "intermediate_size": 256},   # 3 linear layers, ZERO full-attn
        bidirectional_linear=True,
    )
    torch.manual_seed(0)
    model = dllm.pipelines.a2d.A2DQwen3_5LMHeadModel(cfg).eval()
    a = torch.tensor([[101, 102, 103, 104]])
    b = torch.tensor([[101, 102, 103, 999]])  # perturb LAST token
    with torch.no_grad():
        diff = (model(a).logits[:, 0] - model(b).logits[:, 0]).abs().max().item()
    from .common import ERROR_THRESHOLD
    assert diff > ERROR_THRESHOLD, "linear layers not bidirectional"
```

This uses a 3-layer (all linear, no full-attn) model so the only way the last token can affect the first is through bidirectional linear attention.

- [ ] **Step 3: Run**

Run: `pytest scripts/tests/attention/test_invariance_qwen3_5.py -v`
Expected: PASS (both the M1 test and the new bidirectional-linear test).

- [ ] **Step 4: Commit**

```bash
git add dllm/pipelines/a2d/models/qwen3_5/modeling_qwen3_5.py scripts/tests/attention/test_invariance_qwen3_5.py
git commit -m "feat(a2d): enable bidirectional GatedDeltaNet in A2DQwen3_5TextModel"
```

### Task D4: Re-convert + train M2 and compare to M1

- [ ] **Step 1: Re-convert with the flag on**

Re-run convert (the saved a2d config inherits `bidirectional_linear`; set it by editing `.models/a2d/qwen3_5-0.8b/config.json` to add `"bidirectional_linear": true`, or pass through convert). Verify load:

Run: `python -c "import dllm, transformers, torch; m=transformers.AutoModelForMaskedLM.from_pretrained('.models/a2d/qwen3_5-0.8b', dtype=torch.float16); print(m.config.bidirectional_linear)"`
Expected: `True`.

- [ ] **Step 2: Train M2 (same command as C5)** but a longer run matched to M1, into `.../mdlm/tiny-shakespeare-bidir`.

- [ ] **Step 3: Compare** final eval NLL/PPL and sample quality M1 vs M2. Record the delta. If M2 ≤ M1 (no improvement) or unstable, try Approach B (diagonal correction) — see spec §5 — as a follow-up task.

---

## Phase E — Colab orchestration

### Task E1: Colab driver with Drive checkpointing + resume

**Files:**
- Create: `scripts/colab/train_qwen3_5_mdlm.ipynb` (or `.py` driver)

- [ ] **Step 1: Write a driver** that: mounts Drive, runs `setup.sh`, runs extraction + convert (once, cached to Drive), then launches `pt.py` with `--output_dir` on Drive and `--save_steps 0.05`; on restart, detects the latest `checkpoint-*` on Drive and adds `--resume_from_checkpoint <path>`.

- [ ] **Step 2: Verify resume** — kill mid-run, re-run the driver, confirm it resumes from the last Drive checkpoint (step count continues, loss continuous).

- [ ] **Step 3: Commit**

```bash
git add scripts/colab/train_qwen3_5_mdlm.ipynb
git commit -m "feat(colab): T4 driver with Drive checkpoint/resume for a2d-qwen3_5 MDLM"
```

---

## Self-Review

**Spec coverage:**
- Spec §5 M0 (text extraction) → Phase B (B2/B3). ✓
- Spec §5 M1 (full-attn-only bidirectional) → Phase C (C1–C6, gate at C6). ✓
- Spec §5 M2 (bidirectional GatedDeltaNet, Approach A) → Phase D (D1–D4). ✓
- Spec §4 decisions: Base ckpt (B3/C5 use `-Base`), QLoRA (C5 `--load_in_4bit --lora`), fp16+sdpa (C5 flags), gate at M1 (C6). ✓
- Spec §6 training config → C5. ✓ Spec §7 env/transformers risk → A1. ✓ Spec §8 validation → C3/C6/D1/D3. ✓
- Spec §9 risks: mask-token (B2 resize), transformers pin (A1), padding in bidirectional scan (noted in D2/known limitation — SFT padding handling deferred; PT is unpadded). ✓
- Spec §3 dllm integration points → C2. ✓

**Known limitation (carry into execution):** Approach A's bidirectional scan is correct for **unpadded** (packed PT) batches. Right-padded SFT batches will contaminate the backward scan; SFT is out of the initial scope and needs a masked-reverse-scan follow-up before M2 SFT. The `future_affects_past` tests use unpadded inputs and are valid.

**Placeholder scan:** No TBD/TODO; every code step has complete code; every command has expected output. ✓

**Type/name consistency:** `A2DQwen3_5TextConfig` / `A2DQwen3_5TextModel` / `A2DQwen3_5LMHeadModel` / `A2DQwen3_5GatedDeltaNet`, `model_type="a2d-qwen3_5"`, map key `"qwen3_5_text"`, flag `bidirectional_linear`, paths `.models/qwen3_5-0.8b-text` and `.models/a2d/qwen3_5-0.8b` — used consistently across B/C/D. ✓
