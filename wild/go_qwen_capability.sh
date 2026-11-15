#!/usr/bin/env bash
# Capability cost of the released Qwen 3 32B cap (user's go 2026-11-13): the Llama capability run (results/cap,
# 2026-11-03) with AXIS_MODEL=qwen. Same four benchmarks, stage-1 sizes and lm-eval settings (capability.py):
# IFEval 200, MMLU Pro 210 (15 per subject), GSM8K 200, EQ-Bench 171; lm-eval 0.4.9.2, chat template, few-shot
# as turns, greedy. Uncapped, then capped with layers_46:54-p0.25; thinking off. Answers are cached per condition
# and capability.json is saved after each benchmark, so a rerun resumes. Scoring runs on the laptop
# (capability_score.py). Stops the pod when done (after up to 60 min for the laptop to fetch); a backstop stops
# it 4 h after launch regardless. NO_STOP=1: no stop at the end (to run another job next); the backstop stays.
set -uo pipefail
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
export AXIS_MODEL=qwen HF_HOME=/workspace/hf PYTHONPATH=/workspace/repo PATH=/workspace/venv/bin:/root/.local/bin:$PATH
PY=/workspace/venv/bin/python
R=/workspace/wild/qcap
mkdir -p $R
cd /workspace/repo/wild
exec >> /root/qcap.log 2>&1  # local disk: appends to the network volume were being lost
step() { echo "== $(date -u +%FT%TZ) $*"; }
STOP="/usr/bin/runpodctl stop pod $RUNPOD_POD_ID"
step "stop command check"; /usr/bin/runpodctl get pod $RUNPOD_POD_ID || echo "WARNING: runpodctl cannot see this pod; laptop stop is the fallback"
setsid nohup bash -c "sleep 14400; $STOP" > /dev/null 2>&1 < /dev/null &
finish() {
  echo "== $(date -u +%FT%TZ) end"; cp /root/qcap.log $R/qcap.log; touch /workspace/.qcap_done
  [ "${NO_STOP:-0}" = 1 ] && return
  for i in $(seq 180); do [ -f /workspace/.qcap_fetched ] && break; sleep 20; done; $STOP
}
trap finish EXIT
# setup.sh's packages without its Llama 70B download, plus lm-eval at run.sh's pin. The pins repeat so lm-eval's
# install cannot move transformers or vllm (torch comes with vllm).
step setup; command -v uv >/dev/null || pip install -q --break-system-packages uv
[ -x $PY ] || uv venv /workspace/venv --python 3.11
uv pip install -q --python $PY vllm==0.23.0 transformers==4.57.6 accelerate huggingface_hub pyarrow scipy tqdm scikit-learn plotly openai python-dotenv "lm_eval[ifeval]==0.4.9.2" || echo "FAILED setup"
step qwen weights; $PY -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-32B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])"
$PY -c "import wcommon as w; print('config', w.MODEL, w.LAYER, w.AXIS_FILE, w.CAPPING_FILE, w.CAPPING_EXPERIMENT, w.CHAT_KW)"
# Batch size by total GPU memory (an estimate, not measured on Qwen): weights take about 66 GB and MMLU Pro's
# prompts plus 2,048 new tokens about 1.2 GB per sequence. 48 is the Llama run's value (2x H200). Set CAP_BATCH
# to override. Batch size changes speed, not which questions are asked.
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s += $1} END {print s + 0}')
if [ -z "${CAP_BATCH:-}" ]; then
  if [ "$MEM" -ge 170000 ]; then CAP_BATCH=48; elif [ "$MEM" -ge 130000 ]; then CAP_BATCH=32; else CAP_BATCH=8; fi
fi
export CAP_BATCH; echo "GPU memory $MEM MiB, batch $CAP_BATCH"
touch $R/capability_approved  # Benchmark set approved 2026-11-02; this Qwen run approved 2026-11-13. Stage 1 only.
step capability; $PY capability.py $R || echo "FAILED capability"
step done
