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
