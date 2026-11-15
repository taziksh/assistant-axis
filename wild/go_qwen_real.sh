#!/usr/bin/env bash
# Does Qwen's cap (layers_46:54-p0.25, -41% harm on Lu et al.'s persona jailbreaks) also cut harm on real
# attacks (user's go 2026-11-08)? Same code and settings as go_lu2.sh's Qwen run on Lu; only the prompts
# differ: in-the-wild jailbreaks (325) on GPU 1, MHJ human multi-turn (537) on GPU 0, each uncapped then capped.
# Judging runs on the laptop. Stops the pod when done (after up to 30 min for the laptop to fetch);
# backstop stops it 2.5 h after launch regardless.
set -uo pipefail
export RUNPOD_POD_ID="$(tr "\0" "\n" < /proc/1/environ | grep "^RUNPOD_POD_ID=" | cut -d= -f2-)"
export HF_HOME=/dev/shm/hf PYTHONPATH=/workspace/repo PATH=/workspace/venv/bin:/root/.local/bin:$PATH
export CAP_TEMPERATURE=0.7 CAP_TOKEN_BUDGET=60000 CAP_MAX_BATCH=48
export CAP_MODEL=Qwen/Qwen3-32B CAP_FILE=qwen-3-32b/capping_config.pt CAP_EXP=layers_46:54-p0.25
PY=/workspace/venv/bin/python
D=/workspace/wild/lu/qwen_real; mkdir -p $D
cd /workspace/repo/wild
exec >> $D/run.log 2>&1
STOP="/usr/bin/runpodctl stop pod $RUNPOD_POD_ID"
setsid nohup bash -c "sleep 9000; $STOP" > /dev/null 2>&1 < /dev/null &
trap 'echo "== $(date -u +%FT%TZ) end"; touch /workspace/.real_done; for i in $(seq 90); do [ -f /workspace/.real_fetched ] && break; sleep 20; done; '"$STOP" EXIT
step() { echo "== $(date -u +%FT%TZ) $*"; }
[ -x $PY ] && $PY -c "import vllm" 2>/dev/null || { step setup; (cd /workspace/repo && bash sanity_check/scripts/setup.sh); }
$PY -c "import pandas, pyarrow" 2>/dev/null || uv pip install -q --python $PY pandas pyarrow
step qwen weights; $PY -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-32B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])"
run() {  # GPU SET
  for c in none capped; do
    CUDA_VISIBLE_DEVICES=$1 $PY capping.py $D data/dl --only $2:$c > $D/$2_$c.log 2>&1 || echo "FAILED $2:$c"
  done
}
step "capping mhj (gpu 0) and itw (gpu 1)"
run 0 mhj & run 1 itw & wait
step done
