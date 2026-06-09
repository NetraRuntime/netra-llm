"""
Modal H100 training for the a2d-qwen3_5 masked-diffusion adaptation of Qwen3.5-0.8B.

On an H100 (80 GB, native bf16, fast SDPA) we drop every T4 hack: bf16, full
fine-tuning (no QLoRA), sdpa attention, bigger batch + longer context. Checkpoints
persist in a Modal Volume; metrics stream to Weights & Biases (rifky/netra-0.8b).

The LOCAL repo is mounted into the container, so your local edits apply without
re-pushing to GitHub or rebuilding the image.

------------------------------------------------------------------------------
One-time setup (already done if you followed the chat):
    modal profile activate netragratis        # or prefix every command: MODAL_PROFILE=netragratis ...
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
import pathlib

import modal

# Repo root = two levels up from this file (scripts/modal/train_a2d.py).
REPO_LOCAL = str(pathlib.Path(__file__).resolve().parents[2])
REMOTE = "/root/netra-llm"
DATA = "/data"
BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # CUDA-matched torch (H100 = sm_90) — must use the cu124 wheel, not the CPU default.
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
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
    .env(
        {
            "PYTHONPATH": REMOTE,
            "HF_HOME": f"{DATA}/hf_cache",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    # Mount the LOCAL repo last so edits apply without rebuild. .git/caches excluded.
    .add_local_dir(
        REPO_LOCAL,
        REMOTE,
        ignore=[
            "**/.git/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/.data/**",
            "**/.models/**",
            "lm-evaluation-harness/**",
            "assets/**",
            ".venv/**",
        ],
    )
)

app = modal.App("netra-a2d-qwen35", image=image)
vol = modal.Volume.from_name("netra-a2d", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb")

TEXT_DIR = f"{DATA}/qwen3_5-0.8b-text"
A2D_DIR = f"{DATA}/a2d/qwen3_5-0.8b"


def _sh(cmd: str):
    import subprocess

    print("+ " + cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=REMOTE)


@app.function(cpu=8.0, memory=49152, timeout=3 * 3600, volumes={DATA: vol})
def extract_and_convert(force: bool = False):
    """M0 + convert: text-backbone extraction and AR->diffusion conversion (CPU is fine)."""
    import os

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


@app.function(
    gpu="H100",
    timeout=24 * 3600,
    volumes={DATA: vol},
    secrets=[wandb_secret],
)
def train(
    dataset: str = "Trelis/tiny-shakespeare",
    text_field: str = "Text",
    insert_eos: bool = False,
    max_length: int = 512,
    batch: int = 16,
    grad_accum: int = 1,
    epochs: float = 10,
    lr: float = 1e-4,
    lora: bool = False,
    run_name: str = "m1-tinyshake",
):
    """MDLM continual-pretraining on one H100 (bf16, sdpa, full FT by default)."""
    import os

    out = f"{DATA}/runs/{run_name}"
    resume = ""
    if os.path.isdir(out):
        ckpts = [d for d in os.listdir(out) if d.startswith("checkpoint-") and d[11:].isdigit()]
        if ckpts:
            latest = max(ckpts, key=lambda d: int(d[11:]))
            resume = f"--resume_from_checkpoint '{out}/{latest}'"
            print(f"[resume] from {out}/{latest}", flush=True)

    lora_flag = "--lora True" if lora else ""
    _sh(
        "python -u examples/a2d/mdlm/pt.py "
        f"--model_name_or_path '{A2D_DIR}' "
        f"--dataset_args '{dataset}' --text_field '{text_field}' --insert_eos {insert_eos} "
        f"--max_length {max_length} "
        "--dtype bfloat16 --bf16 True --fp16 False --attn_implementation sdpa "
        f"{lora_flag} "
        f"--per_device_train_batch_size {batch} --gradient_accumulation_steps {grad_accum} "
        f"--num_train_epochs {epochs} --learning_rate {lr} --logging_steps 5 "
        "--eval_strategy no --report_to wandb "
        f"--run_name '{run_name}' "
        "--save_strategy steps --save_steps 0.1 --save_total_limit 3 --save_only_model False "
        f"--output_dir '{out}' {resume}"
    )
    vol.commit()


@app.function(gpu="H100", timeout=3600, volumes={DATA: vol})
def sample(run_name: str = "m1-tinyshake", steps: int = 128, max_new_tokens: int = 128):
    """Sample from a trained checkpoint (the M1 gate)."""
    ckpt = f"{DATA}/runs/{run_name}/checkpoint-final"
    _sh(
        "python -u examples/a2d/mdlm/sample.py "
        f"--model_name_or_path '{ckpt}' "
        "--dtype bfloat16 --attn_implementation sdpa "
        f"--steps {steps} --max_new_tokens {max_new_tokens} "
        "--remasking low_confidence --temperature 0.0"
    )


@app.local_entrypoint()
def main(
    dataset: str = "Trelis/tiny-shakespeare",
    text_field: str = "Text",
    max_length: int = 512,
    batch: int = 16,
    epochs: float = 10,
    run_name: str = "m1-tinyshake",
):
    """extract+convert (cached on the Volume) then train, end to end."""
    extract_and_convert.remote()
    train.remote(
        dataset=dataset,
        text_field=text_field,
        max_length=max_length,
        batch=batch,
        epochs=epochs,
        run_name=run_name,
    )
    print(
        "Training launched. Watch W&B: https://wandb.ai/rifky/netra-0.8b\n"
        "Sample/gate: MODAL_PROFILE=netragratis modal run scripts/modal/train_a2d.py::sample "
        f"--run-name {run_name}"
    )
