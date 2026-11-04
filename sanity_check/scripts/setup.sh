#!/usr/bin/env bash
# One-time pod setup: venv, packages, model weights. All repos used are public; an HF_TOKEN
# only helps with rate limits.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf}
command -v uv >/dev/null || pip install -q --break-system-packages uv  # Ubuntu 24.04 images block plain pip (PEP 668).
uv venv /workspace/venv --python 3.11
# Versions used for the 2026-10-31 run (torch 2.11.0+cu130 comes with vllm).
uv pip install --python /workspace/venv/bin/python vllm==0.23.0 transformers==4.57.6 accelerate huggingface_hub pyarrow scipy tqdm scikit-learn plotly openai python-dotenv
/workspace/venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/Llama-3.3-70B-Instruct', allow_patterns=['*.json', '*.safetensors', 'tokenizer*'])"
command -v runpodctl && echo "RUNPOD_POD_ID=${RUNPOD_POD_ID:-unset}"
