#!/usr/bin/env bash
# Why did the released cap not cut harm on Llama (user's go 2026-11-08)? On Lu et al.'s 1,075 jailbreaks:
# 1. Llama: how often the cap binds on our saved replies (capdiag.py).
# 2. Llama: their own capping script (capped/1_capping_vllm.py, vLLM + chatspace at its pinned versions),
#    capped and with a cap too high to bind (same code path, no effect).
# 3. Qwen: the same with their script, then our HF capping (capping.py) and capdiag.py.
# Judging runs on the laptop. Each step may fail without stopping the rest. The pod stops itself when done
# (after up to 60 min for the laptop to fetch); a backstop stops it 4 h after launch regardless.
set -uo pipefail
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
export HF_HOME=/dev/shm/hf PYTHONPATH=/workspace/repo PATH=/workspace/venv/bin:/root/.local/bin:$PATH
export CAP_TEMPERATURE=0.7 CAP_TOKEN_BUDGET=60000 CAP_MAX_BATCH=48
PY=/workspace/venv/bin/python
D=/workspace/wild/lu; mkdir -p $D/theirs $D/qwen
cd /workspace/repo/wild
exec >> $D/run2.log 2>&1
STOP="/usr/bin/runpodctl stop pod $RUNPOD_POD_ID || runpodctl stop pod $RUNPOD_POD_ID"
setsid nohup bash -c "sleep 14400; $STOP" > /dev/null 2>&1 &
trap 'echo "== $(date -u +%FT%TZ) end"; touch /workspace/.lu2_done; for i in $(seq 180); do [ -f /workspace/.lu2_fetched ] && break; sleep 20; done; '"$STOP" EXIT
step() { echo "== $(date -u +%FT%TZ) $*"; }
weights() { $PY -c "from huggingface_hub import snapshot_download as s; s('$1', allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])"; }

# Their package at the commit we read, in its own environment from its lock file (vLLM 0.11.0, torch 2.8).
CS=/root/git/chatspace
(
  command -v uv >/dev/null || pip install -q uv
  [ -d $CS ] || git clone -q https://github.com/GallagherCommaJack/chatspace $CS
  cd $CS && git checkout -q f8037e3 && uv sync --frozen -q && echo "chatspace env ready"
) > $D/theirs/setup.log 2>&1 &
SETUP=$!

# Their config plus "nobind": the same vectors with a cap no token reaches (same code path, no effect).
cfg() {
  $PY - "$1" "$2" "$3" <<'P'
import sys, torch
from huggingface_hub import hf_hub_download
cfg = torch.load(hf_hub_download("lu-christina/assistant-axis-vectors", sys.argv[1], repo_type="dataset"), weights_only=False)
exp = next(e for e in cfg["experiments"] if e["id"] == sys.argv[2])
nobind = {"id": "nobind", "interventions": [{"vector": i["vector"], "cap": 1e6} for i in exp["interventions"]]}
torch.save({"vectors": cfg["vectors"], "experiments": [exp, nobind]}, sys.argv[3])
print("config", sys.argv[3], [e["id"] for e in [exp, nobind]])
P
}
theirs() {  # MODEL EXP OUT EXTRA...
  local model=$1 exp=$2 out=$3; shift 3
  wait $SETUP; grep -q "env ready" $D/theirs/setup.log || { echo "chatspace setup failed"; tail -20 $D/theirs/setup.log; return 1; }
  (cd $CS && env -u PYTHONPATH uv run python /workspace/lu_code/1_capping_vllm.py --config_filepath $D/theirs/$out.pt \
     --prompts_file /workspace/repo/wild/data/lu/theirs_1075.jsonl --output_jsonl $D/theirs/$out.jsonl \
     --model_name $model --max_model_len 4096 --experiment_ids $exp nobind "$@")
}

[ -x $PY ] && $PY -c "import vllm" 2>/dev/null || { step setup; (cd /workspace/repo && bash sanity_check/scripts/setup.sh); }
$PY -c "import pandas, pyarrow" 2>/dev/null || uv pip install -q --python $PY pandas pyarrow
step llama weights; weights unsloth/Llama-3.3-70B-Instruct
step llama capdiag; $PY capdiag.py data/lu/pairs.jsonl $D/cap $D/capdiag_llama.json --n 150 || echo "FAILED llama capdiag"
step llama theirs; cfg llama-3.3-70b/capping_config.pt layers_56:72-p0.25 $D/theirs/llama.pt && \
  theirs unsloth/Llama-3.3-70B-Instruct layers_56:72-p0.25 llama --company Meta --tensor_parallel_size 2 || echo "FAILED llama theirs"
step clear llama; rm -rf /dev/shm/hf/hub/models--unsloth--Llama-3.3-70B-Instruct
step qwen weights; weights Qwen/Qwen3-32B
step qwen theirs; cfg qwen-3-32b/capping_config.pt layers_46:54-p0.25 $D/theirs/qwen.pt && \
  theirs Qwen/Qwen3-32B layers_46:54-p0.25 qwen --company Alibaba --name Qwen --thinking false --tensor_parallel_size 1 || echo "FAILED qwen theirs"
export CAP_MODEL=Qwen/Qwen3-32B CAP_FILE=qwen-3-32b/capping_config.pt CAP_EXP=layers_46:54-p0.25
step qwen capping; $PY capping.py $D/qwen data/dl --shah data/lu/pairs_qwen.jsonl || echo "FAILED qwen capping"
step qwen capdiag; $PY capdiag.py data/lu/pairs_qwen.jsonl $D/qwen/cap $D/capdiag_qwen.json --n 150 || echo "FAILED qwen capdiag"
step done
