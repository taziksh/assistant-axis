#!/usr/bin/env bash
# Paper 4.3 (harm vs drift) on the paper's own Qwen 3 32B openings, with the paper's own kind of second-turn
# questions (user's go 2026-11-17): 430 questions from Lu et al.'s persona-jailbreak set after 550 of the 2,750 role
# openings (q43b_build.py), the authors' own Qwen replies prefilled as turn 1. The openings' projections come from the
# q43 run. Thinking off, temp 0.7, top-p 0.9, 512 tokens (their script's settings). One harm_wild.py per GPU
# (--shard), saving after every 10 openings, so a restart resumes. Judging runs on the laptop. Stops the pod when
# done (after up to 60 min for the laptop to fetch); a backstop stops it 12 h after launch regardless.
set -uo pipefail
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
export AXIS_MODEL=qwen HF_HOME=/workspace/hf PYTHONPATH=/workspace/repo PATH=/workspace/venv/bin:/root/.local/bin:$PATH
PY=/workspace/venv/bin/python
Q=data/dl/chqa/data/catqa_english.json  # unused with --questions-jsonl, but a required argument
R=/workspace/wild/q43b
cd /workspace/repo/wild
exec >> /root/q43b.log 2>&1  # local disk: appends to the network volume were being lost
step() { echo "== $(date -u +%FT%TZ) $*"; }
STOP="/usr/bin/runpodctl stop pod $RUNPOD_POD_ID"
step "stop command check"; /usr/bin/runpodctl get pod $RUNPOD_POD_ID || echo "WARNING: runpodctl cannot see this pod; laptop stop is the fallback"
setsid nohup bash -c "sleep 43200; $STOP" > /dev/null 2>&1 < /dev/null &
trap 'echo "== $(date -u +%FT%TZ) end"; cp /root/q43b*.log /workspace/wild/; touch /workspace/.q43b_done; for i in $(seq 180); do [ -f /workspace/.q43b_fetched ] && break; sleep 20; done; '"$STOP" EXIT
# setup.sh's packages, without its Llama 70B download.
[ -x $PY ] && $PY -c "import vllm" 2>/dev/null || { step setup; command -v uv >/dev/null || pip install -q --break-system-packages uv
  uv venv /workspace/venv --python 3.11; uv pip install --python $PY vllm==0.23.0 transformers==4.57.6 accelerate huggingface_hub pyarrow scipy tqdm scikit-learn plotly openai python-dotenv; }
step qwen weights; $PY -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-32B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])"
$PY -c "import wcommon as w; print('config', w.MODEL, w.LAYER, w.AXIS_FILE, w.CHAT_KW)"
N=$(nvidia-smi -L | wc -l)
step "harm on $N GPUs"
shard() {  # GPU index; up to 3 tries, each resuming from the saved replies.
  for try in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$1 $PY harm_wild.py $R $Q harm --per-band 100000 --keep-truncated \
      --questions-jsonl $R/questions.jsonl --chunk 10 --shard $1/$N --tag .s$1 >> /root/q43b.s$1.log 2>&1 && return 0
    echo "FAILED shard $1 try $try"
  done
  return 1
}
for g in $(seq 0 $((N - 1))); do shard $g & done
wait
grep -h "^harm:" /root/q43b.s*.log | tail -n $((2 * N))
step done
