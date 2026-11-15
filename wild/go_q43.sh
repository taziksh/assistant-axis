#!/usr/bin/env bash
# Paper 4.3 (harm vs drift) on the paper's own Qwen 3 32B openings (user's go 2026-11-12): the 2,750 role openings
# from persona-subspace dynamics/results/qwen-3-32b/prefills/all_roles.jsonl, with the authors' own Qwen replies
# prefilled as turn 1 (as their 3_jailbreak_prefills.py does). Turn 2: 16 HarmBench questions from their pipeline's
# harmbench.jsonl (child-related ones excluded), the same 16 after every opening. Thinking off, temp 0.7, top-p 0.9,
# 512 tokens (their script's settings). activations.py adds our Assistant Axis projection of each opening reply;
# the authors' pc1/contrast projections are kept locally (paper_proj.jsonl). Judging runs on the laptop. Stops the
# pod when done (after up to 60 min for the laptop to fetch); a backstop stops it 4 h after launch regardless.
set -uo pipefail
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
export AXIS_MODEL=qwen HF_HOME=/workspace/hf PYTHONPATH=/workspace/repo PATH=/workspace/venv/bin:/root/.local/bin:$PATH
PY=/workspace/venv/bin/python
Q=data/dl/chqa/data/catqa_english.json  # unused with --questions-jsonl, but a required argument
R=/workspace/wild/q43
cd /workspace/repo/wild
exec >> /root/q43.log 2>&1  # local disk: appends to the network volume were being lost
step() { echo "== $(date -u +%FT%TZ) $*"; }
STOP="/usr/bin/runpodctl stop pod $RUNPOD_POD_ID"
step "stop command check"; /usr/bin/runpodctl get pod $RUNPOD_POD_ID || echo "WARNING: runpodctl cannot see this pod; laptop stop is the fallback"
setsid nohup bash -c "sleep 14400; $STOP" > /dev/null 2>&1 < /dev/null &
trap 'echo "== $(date -u +%FT%TZ) end"; cp /root/q43.log /workspace/wild/q43.log; touch /workspace/.q43_done; for i in $(seq 180); do [ -f /workspace/.q43_fetched ] && break; sleep 20; done; '"$STOP" EXIT
# setup.sh's packages, without its Llama 70B download.
[ -x $PY ] && $PY -c "import vllm" 2>/dev/null || { step setup; command -v uv >/dev/null || pip install -q --break-system-packages uv
  uv venv /workspace/venv --python 3.11; uv pip install --python $PY vllm==0.23.0 transformers==4.57.6 accelerate huggingface_hub pyarrow scipy tqdm scikit-learn plotly openai python-dotenv; }
step qwen weights; $PY -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-32B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])"
$PY -c "import wcommon as w; print('config', w.MODEL, w.LAYER, w.AXIS_FILE, w.CHAT_KW)"
step activations; $PY activations.py $R || echo "FAILED activations"
[ -f $R/harm_wild.jsonl ] || { step harm; $PY harm_wild.py $R $Q harm --per-band 100000 --keep-truncated --questions-jsonl $R/hb16.jsonl || echo "FAILED harm"; }
step done
