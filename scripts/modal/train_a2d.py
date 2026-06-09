"""
Modal H100 training for the a2d-qwen3_5 masked-diffusion adaptation of Qwen3.5-0.8B.

On an H100 (80 GB, native bf16, fast SDPA) we drop every T4 hack: bf16, full
fine-tuning (no QLoRA), sdpa attention, bigger batch + longer context. Checkpoints
persist in a Modal Volume; metrics stream to Weights & Biases (rifky/netra-0.8b).

Deps are baked into a cached image; the repo is cloned fresh at the start of each
function, so pushing to the branch is enough to apply code changes (no image rebuild).

------------------------------------------------------------------------------
One-time setup:
    modal profile activate netragratis      # or prefix every command: MODAL_PROFILE=netragratis ...
    modal secret create wandb \
        WANDB_API_KEY=<key> WANDB_ENTITY=rifky WANDB_PROJECT=netra-0.8b

Full Milestone-1 pipeline (extract -> convert -> train) on one H100:
    MODAL_PROFILE=netragratis modal run scripts/modal/train_a2d.py

Just (re)train, or change knobs:
    MODAL_PROFILE=netragratis modal run scripts/modal/train_a2d.py::train \
        --dataset Trelis/tiny-shakespeare --max-length 512 --batch 16 --epochs 10 \
        --run-name m1-tinyshake

Sample from a checkpoint (the M1 gate):
    MODAL_PROFILE=netragratis modal run scripts/modal/train_a2d.py::sample --run-name m1-tinyshake
------------------------------------------------------------------------------
"""
import modal

GIT_URL = "https://github.com/NetraRuntime/netra-llm.git"
GIT_REF = "a2d-qwen3_5-bidirectional"
REMOTE = "/root/netra-llm"
DATA = "/data"
BASE_MODEL = "Qwen/Qwen3.5-4B-Base"   # real bilingual-run base (0.8B was the gate, already validated)
_SLUG = "qwen3_5-4b"
TEXT_DIR = f"{DATA}/{_SLUG}-text"
A2D_DIR = f"{DATA}/a2d/{_SLUG}"
N_GPU = 8  # B200 count for the multi-GPU bilingual run (train_multi)

# Continued-PT corpus: EN + Bahasa Indonesia + code, interleaved 1/3 each via [weight].
# Code (opc-fineweb-code-corpus, "text" field) adds the structured/JSON/function syntax that
# transfers to agentic/tool-calling (DAPT). [train:N] caps each source to ~250M unique tokens
# so the run does several epochs (diffusion benefits from repetition); EN/ID kept at full weight.
CORPUS = (
    "HuggingFaceFW/fineweb-edu[name:sample-10BT,train:240000,weight:1]"
    "+HuggingFaceFW/fineweb-2[name:ind_Latn,train:620000,weight:1]"
    "+OpenCoder-LLM/opc-fineweb-code-corpus[train:420000,weight:1]"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # B200 = Blackwell (sm_100) → needs a CUDA 12.8 torch build; cu124 has no Blackwell kernels.
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    # dllm runtime deps (minus deepspeed/bitsandbytes — not needed for single-GPU full FT).
    .pip_install(
        "accelerate",
        "peft>=0.19.1",
        "datasets",
        "sentencepiece",
        "torchmetrics",
        "tyro",
        "omegaconf",
        "tqdm",
        "rich",
        "wandb",
        "lm_eval",
        "trl",
        "hf_transfer",
    )
    # transformers main = the build that ships qwen3_5.
    .run_commands("pip install --upgrade 'git+https://github.com/huggingface/transformers.git'")
    # flash-linear-attention = Triton gated-delta-net kernels (delta-net fast path; torch fallback
    # is ~23s/step at 4B). Do NOT install tilelang: fla 0.5.0 routes gated chunk_bwd to the tilelang
    # backend when it's present, and tilelang can't find a CUDA toolkit on this image ("No CUDA
    # available"). The fla Triton-kernel bug is Hopper(H100)-specific; on Blackwell(B200) the Triton
    # backend is correct, so fla-only gives the fast path. Best-effort import → torch fallback.
    .run_commands("pip install flash-linear-attention || echo 'fla unavailable'")
    .env(
        {
            "PYTHONPATH": REMOTE,
            "HF_HOME": f"{DATA}/hf_cache",
            "TOKENIZERS_PARALLELISM": "false",
            # the gated delta-net torch fallback runs in fp32 across 18 layers; reduce
            # allocator fragmentation so big transient tensors fit.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

app = modal.App("netra-a2d-qwen35", image=image)
vol = modal.Volume.from_name("netra-a2d", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb")


def _ensure_repo():
    """Clone (or fast-forward) the branch into the ephemeral container FS at runtime."""
    import os
    import subprocess

    if os.path.isdir(f"{REMOTE}/.git"):
        subprocess.run(
            f"git -C {REMOTE} fetch --depth 1 origin {GIT_REF} "
            f"&& git -C {REMOTE} reset --hard FETCH_HEAD",
            shell=True,
            check=True,
        )
    else:
        subprocess.run(
            f"git clone --depth 1 -b {GIT_REF} {GIT_URL} {REMOTE}", shell=True, check=True
        )


def _sh(cmd: str):
    import subprocess

    print("+ " + cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=REMOTE)


@app.function(cpu=8.0, memory=98304, timeout=4 * 3600, volumes={DATA: vol})
def extract_and_convert(force: bool = False):
    """M0 + convert: text-backbone extraction and AR->diffusion conversion (CPU is fine)."""
    import os

    _ensure_repo()
    if force or not os.path.exists(f"{TEXT_DIR}/config.json"):
        _sh(
            f"python dllm/tools/extract_qwen3_5_text.py "
            f"--model_name_or_path '{BASE_MODEL}' --output_dir '{TEXT_DIR}'"
        )
    else:
        print(f"[skip] text backbone already at {TEXT_DIR}")
    if force or not os.path.exists(f"{A2D_DIR}/config.json"):
        _sh(
            f"python dllm/pipelines/a2d/convert.py "
            f"--model_name_or_path '{TEXT_DIR}' --output_dir '{A2D_DIR}'"
        )
    else:
        print(f"[skip] a2d model already at {A2D_DIR}")
    vol.commit()


@app.function(gpu="H100", timeout=24 * 3600, volumes={DATA: vol}, secrets=[wandb_secret])
def train(
    dataset: str = "Trelis/tiny-shakespeare",
    text_field: str = "Text",
    insert_eos: bool = False,
    max_length: int = 512,
    batch: int = 8,
    grad_accum: int = 1,
    epochs: int = 10,
    lr: float = 1e-4,
    block_size: int = 32,
    model_dir: str = A2D_DIR,
    run_name: str = "bd3lm-tinyshake",
):
    """BD3LM (block-diffusion) continual-pretraining on one H100 (bf16, sdpa) — gate/smoke."""
    import os

    _ensure_repo()
    out = f"{DATA}/runs/{run_name}"
    resume = ""
    if os.path.isdir(out):
        ckpts = [d for d in os.listdir(out) if d.startswith("checkpoint-") and d[11:].isdigit()]
        if ckpts:
            latest = max(ckpts, key=lambda d: int(d[11:]))
            resume = f"--resume_from_checkpoint '{out}/{latest}'"
            print(f"[resume] from {out}/{latest}", flush=True)

    _sh(
        "python -u examples/a2d/bd3lm/pt.py "
        f"--model_name_or_path '{model_dir}' "
        f"--dataset_args '{dataset}' --text_field '{text_field}' --insert_eos {insert_eos} "
        f"--max_length {max_length} --block_size {block_size} "
        "--dtype bfloat16 --bf16 True --fp16 False --attn_implementation sdpa "
        "--gradient_checkpointing True "
        f"--per_device_train_batch_size {batch} --gradient_accumulation_steps {grad_accum} "
        f"--num_train_epochs {epochs} --learning_rate {lr} --logging_steps 5 "
        "--eval_strategy no --report_to wandb "
        f"--run_name '{run_name}' "
        "--save_strategy steps --save_steps 0.1 --save_total_limit 3 --save_only_model False "
        f"--output_dir '{out}' {resume}"
    )
    vol.commit()


PEEK_SCRIPT = r'''
import sys, dllm
dataset, text_field, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ds = dllm.data.load_pt_dataset(dataset, streaming=True)
it = iter(ds["train"])
for i in range(n):
    print(f"[{i}] {next(it)[text_field][:140]!r}", flush=True)
'''


@app.function(cpu=4.0, memory=16384, timeout=1800, volumes={DATA: vol})
def peek_data(dataset: str = CORPUS, text_field: str = "text", n: int = 12):
    """Stream a few examples from a (possibly interleaved) mix to verify it on Modal's fast
    network — confirm EN and ID alternate before spending GPU. Runs as a subprocess so dllm
    imports in a fresh process after the runtime clone (PYTHONPATH was cached empty at startup)."""
    import subprocess

    _ensure_repo()
    subprocess.run(
        ["python", "-u", "-c", PEEK_SCRIPT, dataset, text_field, str(n)],
        cwd=REMOTE,
        check=True,
    )


@app.function(gpu=f"B200:{N_GPU}", timeout=24 * 3600, volumes={DATA: vol}, secrets=[wandb_secret])
def train_multi(
    dataset: str = CORPUS,
    text_field: str = "text",
    max_length: int = 1024,
    batch: int = 8,
    grad_accum: int = 2,
    max_steps: int = 75000,  # eff-batch 128 x 1024 = ~9.8B tok; 1/3 each EN/ID/code (~3.3B, ~13 epochs)
    lr: float = 1e-4,
    block_size: int = 32,
    save_steps: float = 0.05,
    model_dir: str = A2D_DIR,
    run_name: str = "bd3lm-en-id",
):
    """Multi-GPU (N_GPU x B200) BD3LM training on a streamed bilingual corpus, DDP via accelerate."""
    import os

    import torch

    _ensure_repo()
    nproc = torch.cuda.device_count()
    out = f"{DATA}/runs/{run_name}"
    resume = ""
    if os.path.isdir(out):
        ckpts = [d for d in os.listdir(out) if d.startswith("checkpoint-") and d[11:].isdigit()]
        if ckpts:
            latest = max(ckpts, key=lambda d: int(d[11:]))
            resume = f"--resume_from_checkpoint '{out}/{latest}'"
            print(f"[resume] from {out}/{latest}", flush=True)

    _sh(
        f"accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes {nproc} "
        "examples/a2d/bd3lm/pt.py "
        f"--model_name_or_path '{model_dir}' "
        f"--dataset_args '{dataset}' --text_field '{text_field}' --insert_eos True --streaming True "
        f"--max_length {max_length} --block_size {block_size} "
        "--dtype bfloat16 --bf16 True --fp16 False --attn_implementation sdpa "
        "--gradient_checkpointing True "
        f"--per_device_train_batch_size {batch} --gradient_accumulation_steps {grad_accum} "
        f"--max_steps {max_steps} --learning_rate {lr} --logging_steps 10 "
        "--eval_strategy no --report_to wandb "
        f"--run_name '{run_name}' "
        f"--save_strategy steps --save_steps {save_steps} --save_total_limit 3 --save_only_model False "
        f"--output_dir '{out}' {resume}"
    )
    vol.commit()


# Raw / unconditional BD3LM sampling for a *pretrained* (non-instruct) checkpoint.
# (examples/a2d/bd3lm/sample.py applies a chat template with math/code prompts, meaningless
#  for a raw text PT model — here we use empty + text-prefix prompts.)
RAW_SAMPLE = r'''
import sys, dllm
ckpt, steps, mnt, temp, blk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
model = dllm.utils.get_model(model_name_or_path=ckpt, dtype="bfloat16").eval()
tok = dllm.utils.get_tokenizer(model_name_or_path=ckpt)
sampler = dllm.core.samplers.BD3LMSampler(model=model, tokenizer=tok)
cfg = dllm.core.samplers.BD3LMSamplerConfig(
    steps=steps, max_new_tokens=mnt, block_size=blk, temperature=temp, remasking="low_confidence")
prompts = ["", "ROMEO:", "To be, or not to be,", "KING:\n"]
inputs = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
out = sampler.sample(inputs, cfg, return_dict=True)
seqs = dllm.utils.sample_trim(tok, out.sequences.tolist(), inputs)
for p, s in zip(prompts, seqs):
    print("=" * 70); print("PROMPT:", repr(p)); print((s.strip() or "<empty>"))
'''


@app.function(gpu="H100", timeout=3600, volumes={DATA: vol})
def sample(run_name: str = "bd3lm-tinyshake", steps: int = 128, max_new_tokens: int = 128,
           temperature: float = 0.0, block_size: int = 32):
    """Raw/unconditional BD3LM sampling from a trained PT checkpoint (the gate)."""
    import subprocess

    _ensure_repo()
    ckpt = f"{DATA}/runs/{run_name}/checkpoint-final"
    subprocess.run(
        ["python", "-u", "-c", RAW_SAMPLE, ckpt, str(steps), str(max_new_tokens),
         str(temperature), str(block_size)],
        cwd=REMOTE,
        check=True,
    )


@app.local_entrypoint()
def main(
    dataset: str = "Trelis/tiny-shakespeare",
    text_field: str = "Text",
    max_length: int = 512,
    batch: int = 8,
    epochs: int = 10,
    block_size: int = 32,
    model_dir: str = A2D_DIR,
    run_name: str = "bd3lm-tinyshake",
):
    """extract+convert (cached on the Volume) then BD3LM train, end to end."""
    extract_and_convert.remote()
    train.remote(
        dataset=dataset,
        text_field=text_field,
        max_length=max_length,
        batch=batch,
        epochs=epochs,
        block_size=block_size,
        model_dir=model_dir,
        run_name=run_name,
    )
    print(
        "Training launched. Watch W&B: https://wandb.ai/rifky/netra-0.8b\n"
        "Sample/gate: MODAL_PROFILE=netragratis modal run scripts/modal/train_a2d.py::sample "
        f"--run-name {run_name}"
    )
