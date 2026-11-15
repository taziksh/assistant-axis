#!/usr/bin/env bash
# Context-length control for drift (user's go 2026-11-12). Usage: go_lenctrl.sh llama|qwen
# Targets: the 300 drift_turns WildChat chats. At each user turn k (2-12) the model replies to the target's own
# message k after (a) k-1 unrelated exchanges from other WildChat chats (filler) or (b) no history (nohist); built
# by lenctrl_build.py. The real-history condition is drift_turns (Llama, already run); for Qwen it is run here too
# (real). Same generation settings as drift_turns (temp 0.7, top-p 0.9, 1,024 tokens), then the Assistant Axis
# projection of each reply (activations.py). Judging: none. Stops the pod when done (after up to 60 min for the
# laptop to fetch); a backstop stops it 4 h after launch regardless.
set -uo pipefail
M=$1
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
export PYTHONPATH=/workspace/repo PATH=/workspace/venv/bin:/root/.local/bin:$PATH
if [ "$M" = qwen ]; then export AXIS_MODEL=qwen HF_HOME=/workspace/hf; W=Qwen/Qwen3-32B; PAT="['*.json', '*.safetensors', '*.txt', '*.jinja']"
else export HF_HOME=/dev/shm/hf; W=unsloth/Llama-3.3-70B-Instruct; PAT="['*.json', '*.safetensors', 'tokenizer*']"; fi
PY=/workspace/venv/bin/python
Q=data/dl/chqa/data/catqa_english.json  # required argument; pool doesn't read it
R=/workspace/wild/lc
cd /workspace/repo/wild
exec >> /root/lc.log 2>&1  # local disk: appends to the network volume were being lost
step() { echo "== $(date -u +%FT%TZ) $*"; }
STOP="/usr/bin/runpodctl stop pod $RUNPOD_POD_ID"
step "stop command check ($M)"; /usr/bin/runpodctl get pod $RUNPOD_POD_ID || echo "WARNING: runpodctl cannot see this pod; laptop stop is the fallback"
setsid nohup bash -c "sleep 14400; $STOP" > /dev/null 2>&1 < /dev/null &
trap 'echo "== $(date -u +%FT%TZ) end"; cp /root/lc.log $R/lc.log; touch /workspace/.lc_done; for i in $(seq 180); do [ -f /workspace/.lc_fetched ] && break; sleep 20; done; '"$STOP" EXIT
# setup.sh's packages, without its Llama 70B download (weights below).
[ -x $PY ] && $PY -c "import vllm" 2>/dev/null || { step setup; command -v uv >/dev/null || pip install -q --break-system-packages uv
  uv venv /workspace/venv --python 3.11; uv pip install --python $PY vllm==0.23.0 transformers==4.57.6 accelerate huggingface_hub pyarrow scipy tqdm scikit-learn plotly openai python-dotenv; }
step weights; $PY -c "from huggingface_hub import snapshot_download as s; s('$W', allow_patterns=$PAT)"
$PY -c "import wcommon as w; print('config', w.MODEL, w.LAYER, w.AXIS_FILE, w.CHAT_KW)"
CONDS="filler nohist"; [ "$M" = qwen ] && CONDS="real filler nohist"
for c in $CONDS; do
  F=--last-turn; [ $c = real ] && F=--all-turns
  [ -f $R/$c/drift.jsonl ] || { step "$c pool"; $PY harm_wild.py $R/$c $Q pool $F || echo "FAILED $c pool"; }
  step "$c activations"; $PY activations.py $R/$c || echo "FAILED $c activations"
done
step done
