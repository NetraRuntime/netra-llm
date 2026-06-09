#!/usr/bin/env bash
# T4-friendly MDLM training driver for the a2d-qwen3_5 model, with Google-Drive
# checkpoint/resume across Colab disconnects.
#
# Usage:
#   bash scripts/colab/train_qwen3_5_mdlm.sh <A2D_MODEL_DIR> <OUTPUT_DIR> [EPOCHS]
#
# - <A2D_MODEL_DIR>: the converted a2d checkpoint (output of convert.py)
# - <OUTPUT_DIR>:    where to write checkpoints (put this on Drive, e.g.
#                    /content/drive/MyDrive/a2d-qwen3_5/runs/m1)
# Re-run the SAME command after a disconnect: it auto-resumes from the latest
# numeric checkpoint in OUTPUT_DIR (optimizer state included).
set -euo pipefail

MODEL="${1:?need A2D_MODEL_DIR}"
OUTDIR="${2:?need OUTPUT_DIR}"
EPOCHS="${3:-10}"

RESUME=""
# Resume from the highest-step checkpoint-<N> if any exist (ignore checkpoint-final).
if compgen -G "${OUTDIR}/checkpoint-[0-9]*" > /dev/null; then
  LATEST="$(ls -d "${OUTDIR}"/checkpoint-[0-9]* | sed 's/.*checkpoint-//' | sort -n | tail -1)"
  RESUME="--resume_from_checkpoint ${OUTDIR}/checkpoint-${LATEST}"
  echo "[driver] Resuming from ${OUTDIR}/checkpoint-${LATEST}"
else
  echo "[driver] Fresh run into ${OUTDIR}"
fi

accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/a2d/mdlm/pt.py \
    --model_name_or_path "${MODEL}" \
    --dataset_args "Trelis/tiny-shakespeare" --text_field "Text" --insert_eos False \
    --max_length 128 \
    --dtype float16 --bf16 False --fp16 True --attn_implementation sdpa \
    --load_in_4bit True --lora True \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --num_train_epochs "${EPOCHS}" --learning_rate 1e-4 \
    --eval_strategy no --report_to none \
    --save_strategy steps --save_steps 0.05 --save_total_limit 2 \
    --save_only_model False \
    --output_dir "${OUTDIR}" ${RESUME}
