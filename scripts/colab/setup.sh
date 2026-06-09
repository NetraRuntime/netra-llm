#!/usr/bin/env bash
set -euo pipefail
# dllm deps first
pip install -e .
# dllm pins peft==0.17.1, which imports the now-removed top-level `HybridCache` from
# transformers-main and crashes at import time. peft>=0.19.1 made that import lazy.
pip install -U "peft>=0.19.1"
# dllm eagerly imports these at package load (eval harness + RL trainer). The
# lm-evaluation-harness is normally a git submodule we don't need for a2d, so just
# pull the PyPI packages to satisfy `import dllm`.
pip install lm_eval trl
# transformers from git main = the build that actually ships qwen3_5. force-reinstall so
# it wins over any PyPI dev wheel that LACKS the module (a same-versioned wheel exists),
# and keep deps so huggingface_hub etc. upgrade to match. DO NOT run any other
# `pip install transformers...` after this line, or you'll clobber qwen3_5.
pip install --upgrade --force-reinstall "git+https://github.com/huggingface/transformers.git"
# QLoRA deps (optional extra in dllm)
pip install bitsandbytes==0.48.1
# T4: NO flash-attn (sm_75 unsupported). sdpa is used instead.
# (transformers-main dropped top-level HybridCache + is_torch_fx_available; handled by
#  peft>=0.19.1 above and a guarded import in dllm/pipelines/llada2 — no runtime shim.)
python - <<'PY'
import torch
import dllm                  # imports the full package (eval, pipelines, utils)
import dllm.pipelines.a2d    # registers the a2d-qwen3_5 model
import transformers
from transformers import Qwen3_5ForCausalLM
print("transformers", transformers.__version__, "torch", torch.__version__)
print("qwen3_5 + dllm import OK")
PY
