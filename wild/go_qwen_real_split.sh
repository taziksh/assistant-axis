#!/usr/bin/env bash
# Takes over go_qwen_real.sh so MHJ's capped condition runs on GPU 1 once ITW frees it, instead of after
# MHJ uncapped on GPU 0 (saves ~45 min). Keeps the running python jobs; stops only the bash that would
# have queued mhj:capped on GPU 0. Same stop logic as go_qwen_real.sh; its 2.5 h backstop still applies.
set -uo pipefail
export RUNPOD_POD_ID="$(tr "\0" "\n" < /proc/1/environ | grep "^RUNPOD_POD_ID=" | cut -d= -f2-)"
export HF_HOME=/dev/shm/hf PYTHONPATH=/workspace/repo PATH=/workspace/venv/bin:/root/.local/bin:$PATH
export CAP_TEMPERATURE=0.7 CAP_TOKEN_BUDGET=60000 CAP_MAX_BATCH=48
export CAP_MODEL=Qwen/Qwen3-32B CAP_FILE=qwen-3-32b/capping_config.pt CAP_EXP=layers_46:54-p0.25
PY=/workspace/venv/bin/python
D=/workspace/wild/lu/qwen_real
cd /workspace/repo/wild
exec >> $D/run.log 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
C=$(pgrep -f "capping.py $D data/dl --only mhj:none")   # MHJ uncapped python, keep
B=$(ps -o ppid= -p $C | tr -d ' ')                       # its bash loop (would queue mhj:capped)
A=$(ps -o ppid= -p $B | tr -d ' ')                       # go_qwen_real.sh main (trap would stop pod early)
I=$(pgrep -f "capping.py $D data/dl --only itw:" | head -1)
ITWLOOP=$(ps -o ppid= -p $I | tr -d ' ')                  # ITW loop: itw:none then itw:capped, keep
step "split: mhj:none py=$C loop=$B main=$A itw loop=$ITWLOOP"
[ -n "$C" ] && [ -n "$B" ] && [ -n "$A" ] && [ -n "$ITWLOOP" ] || { step "FAILED to find pids, leaving go_qwen_real.sh as is"; exit 0; }
kill -9 $A $B
trap 'echo "== $(date -u +%FT%TZ) end"; touch /workspace/.real_done; for i in $(seq 90); do [ -f /workspace/.real_fetched ] && break; sleep 20; done; /usr/bin/runpodctl stop pod $RUNPOD_POD_ID' EXIT
while kill -0 $ITWLOOP 2>/dev/null; do sleep 15; done
step "itw done; mhj:capped on gpu 1"
CUDA_VISIBLE_DEVICES=1 $PY capping.py $D data/dl --only mhj:capped > $D/mhj_capped.log 2>&1 || echo "FAILED mhj:capped"
while kill -0 $C 2>/dev/null; do sleep 15; done
[ -f $D/cap/mhj_none.jsonl ] || echo "FAILED mhj:none (no output)"
step done
