#!/usr/bin/env bash
# Lu et al.'s persona jailbreak prompts (minus one excluded category), user's go 2026-11-06:
# Llama 3.3 70B uncapped and capped at their settings, then Fig. 11 on Qwen 3 32B. Judging runs on the laptop.
# The pod stops itself when done (after up to 30 min for the laptop to fetch) or on any failure; a separate
# backstop stops it 3.5 h after launch regardless.
set -uo pipefail
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
export HF_HOME=/dev/shm/hf PYTHONPATH=/workspace/repo PATH=/workspace/venv/bin:$PATH
export CAP_TEMPERATURE=0.7 CAP_TOKEN_BUDGET=60000 CAP_MAX_BATCH=48
PY=/workspace/venv/bin/python
D=/workspace/wild/lu; mkdir -p $D
cd /workspace/repo/wild
exec >> $D/run.log 2>&1
trap 'echo "== $(date -u +%FT%TZ) end"; touch /workspace/.lu_done; for i in $(seq 90); do [ -f /workspace/.lu_fetched ] && break; sleep 20; done; /usr/bin/runpodctl stop pod $RUNPOD_POD_ID || runpodctl stop pod $RUNPOD_POD_ID' EXIT
step() { echo "== $(date -u +%FT%TZ) $*"; }
[ -x $PY ] && $PY -c "import vllm" 2>/dev/null || { step setup; (cd /workspace/repo && bash sanity_check/scripts/setup.sh); }
step llama weights; $PY -c "from huggingface_hub import snapshot_download as s; s('unsloth/Llama-3.3-70B-Instruct', allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'])"
step llama capping; $PY capping.py $D data/dl --shah data/lu/pairs.jsonl
step clear weights; rm -rf /dev/shm/hf/hub/models--unsloth--Llama-3.3-70B-Instruct
step qwen weights; $PY -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-32B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])"
step fig11; $PY fig11.py data/lu/fig11 $D/fig11.jsonl
step done
