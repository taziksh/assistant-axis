#!/usr/bin/env bash
# How often the released cap would bind on ordinary chats, by turn (user's go 2026-11-13). Usage: go_capfire.sh llama|qwen
# Inputs (capfire_inputs_<model>.tgz from capfire_inputs.sh, unpacked in /workspace/wild/capfire/<model>): the saved
# drift replies of the length-control runs, {real,filler,nohist}/{selected,drift}.jsonl (real = drift_turns for
# Llama, lc_qwen/real for Qwen), and jb/{pairs,shah_none}.jsonl, the paper's jailbreaks with our saved uncapped
# replies (the reference: capdiag.py measured 97% of reply tokens past the cap per layer on Llama, 99% on Qwen).
# capfire.py teacher-forces each saved reply after its history with the cap off and counts, at every capped layer,
# the reply tokens past the cap and by how much. No generation, no judging. Each unit is appended as it finishes,
# so a rerun resumes. Stops the pod when done (after up to 60 min for the laptop to fetch); a backstop stops it
# 4 h after launch regardless. NO_STOP=1: no stop at the end (to run another job next); the backstop stays.
set -uo pipefail
M=$1
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
export PYTHONPATH=/workspace/repo PATH=/workspace/venv/bin:/root/.local/bin:$PATH
if [ "$M" = qwen ]; then export AXIS_MODEL=qwen HF_HOME=/workspace/hf; W=Qwen/Qwen3-32B; PAT="['*.json', '*.safetensors', '*.txt', '*.jinja']"
else export HF_HOME=/dev/shm/hf; W=unsloth/Llama-3.3-70B-Instruct; PAT="['*.json', '*.safetensors', 'tokenizer*']"; fi
PY=/workspace/venv/bin/python
R=/workspace/wild/capfire/$M
cd /workspace/repo/wild
exec >> /root/capfire.log 2>&1  # local disk: appends to the network volume were being lost
step() { echo "== $(date -u +%FT%TZ) $*"; }
STOP="/usr/bin/runpodctl stop pod $RUNPOD_POD_ID"
step "stop command check ($M)"; /usr/bin/runpodctl get pod $RUNPOD_POD_ID || echo "WARNING: runpodctl cannot see this pod; laptop stop is the fallback"
setsid nohup bash -c "sleep 14400; $STOP" > /dev/null 2>&1 < /dev/null &
finish() {
  echo "== $(date -u +%FT%TZ) end"; mkdir -p $R; cp /root/capfire.log $R/capfire.log; touch /workspace/.capfire_done
  [ "${NO_STOP:-0}" = 1 ] && return
  [ "${NOTHING_TO_FETCH:-0}" = 1 ] && { $STOP; return; }
  for i in $(seq 180); do [ -f /workspace/.capfire_fetched ] && break; sleep 20; done; $STOP
}
trap finish EXIT
for c in real filler nohist; do [ -f $R/$c/drift.jsonl ] && [ -f $R/$c/selected.jsonl ] || { echo "FAILED inputs: $R/$c is missing drift.jsonl or selected.jsonl"; NOTHING_TO_FETCH=1; exit 1; }; done
# setup.sh's packages, without its Llama 70B download (weights below).
[ -x $PY ] && $PY -c "import vllm" 2>/dev/null || { step setup; command -v uv >/dev/null || pip install -q --break-system-packages uv
  uv venv /workspace/venv --python 3.11; uv pip install --python $PY vllm==0.23.0 transformers==4.57.6 accelerate huggingface_hub pyarrow scipy tqdm scikit-learn plotly openai python-dotenv; }
step weights; $PY -c "from huggingface_hub import snapshot_download as s; s('$W', allow_patterns=$PAT)"
$PY -c "import wcommon as w; print('config', w.MODEL, w.LAYER, w.AXIS_FILE, w.CAPPING_FILE, w.CAPPING_EXPERIMENT, w.CHAT_KW)"
# The jailbreak reference first: 150 short units, and its result is known from capdiag.py, so a wrong setup shows
# in the first minutes.
if [ -f $R/jb/pairs.jsonl ] && [ -f $R/jb/shah_none.jsonl ]; then
  step "jb reference"; $PY capfire.py jb $R/jb/pairs.jsonl $R/jb/shah_none.jsonl $R/jb/capfire.jsonl --n 150 || echo "FAILED jb"
  $PY capfire_score.py $R --jb-only || true
else echo "no jb inputs in $R/jb: reference skipped"; fi
for c in real nohist filler; do
  step "$c"; $PY capfire.py chats $R/$c || echo "FAILED $c"
done
step done
