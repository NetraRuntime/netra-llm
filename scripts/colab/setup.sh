#!/usr/bin/env bash
set -euo pipefail
# dllm deps first
pip install -e .
# dllm eagerly imports these at package load (eval harness + RL trainer). The
# lm-evaluation-harness is normally a git submodule we don't need for a2d, so just
# pull the PyPI packages to satisfy `import dllm`.
pip install lm_eval trl
# transformers build that ships qwen3_5 (overrides the relaxed pin)
pip install --upgrade "git+https://github.com/huggingface/transformers.git"
# QLoRA deps (optional extra in dllm)
pip install bitsandbytes==0.48.1
# T4: NO flash-attn (sm_75 unsupported). sdpa is used instead.
python - <<'PY'
import transformers, torch
from transformers import Qwen3_5ForCausalLM
import dllm  # fail loudly here if the package import chain is broken
import dllm.pipelines.a2d  # noqa: F401  (registers the a2d-qwen3_5 model)
print("transformers", transformers.__version__, "torch", torch.__version__)
print("qwen3_5 + dllm import OK")
PY
