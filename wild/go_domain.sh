#!/usr/bin/env bash
# Drift by domain with the context-length controls (user's go 2026-11-14). Usage: go_domain.sh llama|qwen
# Targets (dom_inputs.tgz, unpacked in /root/w/wild/dom): 435 real WildChat chats with 5+ user turns that the
# Llama 3.3 70B classifier labelled coding (100), writing (100), therapy (151) or AI philosophy (84);
# real/selected.jsonl carries the label in "domain". Same three conditions and settings as go_lenctrl.sh: at each
# user turn (1-12) the model replies after the real history (real, --all-turns); at turns 2-12 it also replies to
# the same message after k-1 unrelated exchanges (filler) and with no history (nohist), built by lenctrl_build.py.
# One reply per turn (temp 0.7, top-p 0.9, 1,024 tokens), then the Assistant Axis projection of each reply
# (activations.py). Judging: none.
# Everything runs on the pod's local disk (/root/w). After each condition the results are copied to the network
# volume (/workspace/dom_out/<model>), so they survive the pod. At the end the pod waits up to 10 min for the
# laptop to fetch, then stops itself. A backstop stops it 7 h (llama) or 5 h (qwen) after launch regardless.
set -uo pipefail
M=$1
B=/root/w
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
export PYTHONPATH=$B/repo PATH=$B/venv/bin:/root/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=0
if [ "$M" = qwen ]; then export AXIS_MODEL=qwen HF_HOME=$B/hf; BACK=18000; W=Qwen/Qwen3-32B; PAT="['*.json', '*.safetensors', '*.txt', '*.jinja']"
else
  # Llama's 141 GB of weights go in RAM when /dev/shm has room (faster loads), else on the local disk.
  SHM=$(df --output=avail -BG /dev/shm | tail -1 | tr -dc 0-9)
  if [ "${SHM:-0}" -ge 160 ]; then export HF_HOME=/dev/shm/hf; else export HF_HOME=$B/hf; fi
  BACK=25200; W=unsloth/Llama-3.3-70B-Instruct; PAT="['*.json', '*.safetensors', 'tokenizer*']"; fi
PY=$B/venv/bin/python
Q=data/dl/chqa/data/catqa_english.json  # required argument; pool doesn't read it
R=$B/wild/dom
OUT=/workspace/dom_out/$M
cd $B/repo/wild
exec >> /root/dom.log 2>&1  # local disk: appends to the network volume were being lost
step() { echo "== $(date -u +%FT%TZ) $*"; }
save() { mkdir -p $OUT; cp -r "$@" $OUT/ 2>&1; cp /root/dom.log $OUT/dom.log; sync; }
STOP="/usr/bin/runpodctl stop pod $RUNPOD_POD_ID"
step "stop command check ($M), HF_HOME=$HF_HOME"; /usr/bin/runpodctl get pod $RUNPOD_POD_ID || echo "WARNING: runpodctl cannot see this pod; laptop stop is the fallback"
setsid nohup bash -c "sleep $BACK; $STOP" > /dev/null 2>&1 < /dev/null &
trap 'echo "== $(date -u +%FT%TZ) end"; save; touch $OUT/.done $B/.dom_done; for i in $(seq 30); do [ -f $B/.dom_fetched ] && break; sleep 20; done; '"$STOP" EXIT
# setup.sh's packages, without its Llama 70B download (weights below).
[ -x $PY ] && $PY -c "import vllm" 2>/dev/null || { step setup; command -v uv >/dev/null || pip install -q --break-system-packages uv
  uv venv $B/venv --python 3.11; uv pip install --python $PY vllm==0.23.0 transformers==4.57.6 accelerate huggingface_hub pyarrow scipy tqdm scikit-learn plotly openai python-dotenv; }
step weights; $PY -c "from huggingface_hub import snapshot_download as s; s('$W', allow_patterns=$PAT)"
$PY -c "import wcommon as w; print('config', w.MODEL, w.LAYER, w.AXIS_FILE, w.CHAT_KW)"
for c in real filler nohist; do
  F=--last-turn; [ $c = real ] && F=--all-turns
  [ -f $R/$c/drift.jsonl ] || { step "$c pool"; $PY harm_wild.py $R/$c $Q pool $F || echo "FAILED $c pool"; }
  save $R/$c
  step "$c activations"; $PY activations.py $R/$c || echo "FAILED $c activations"
  save $R/$c
done
step done
