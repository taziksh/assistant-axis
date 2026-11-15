#!/usr/bin/env bash
# Starter sweep (user's go 2026-11-08): does any released Llama setting reproduce Lu et al.'s own table?
# Four role_trait settings, run with their script after go_lu2.sh finishes; baseline is go_lu2's "nobind".
# Rule set before the run: if their strongest (layers_0:40-p0.01, -81.3% in their table) cuts harm by less
# than 40%, the released Llama file does not reproduce their table.
# Takes over from go_lu2.sh's end-of-run stop. Stops the pod when done (after up to 45 min for the laptop to
# fetch); the 4:30 am Toronto backstop set earlier still applies.
set -uo pipefail
for v in RUNPOD_POD_ID; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
export HF_HOME=/dev/shm/hf PATH=/workspace/venv/bin:/root/.local/bin:$PATH
PY=/workspace/venv/bin/python
D=/workspace/wild/lu; CS=/root/git/chatspace
exec >> $D/sweep.log 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
step waiting for go_lu2; until [ -f /workspace/.lu2_done ]; do sleep 15; done
pkill -f "go_lu2.sh" && step "took over go_lu2's stop"
trap 'echo "== $(date -u +%FT%TZ) end"; touch /workspace/.sweep_done; for i in $(seq 135); do [ -f /workspace/.sweep_fetched ] && break; sleep 20; done; /usr/bin/runpodctl stop pod $RUNPOD_POD_ID' EXIT
EXPS="layers_0:40-p0.01 layers_0:80-p0.01 layers_56:72-p0.01 layers_52:68-p0.01"
step llama weights; $PY -c "from huggingface_hub import snapshot_download as s; s('unsloth/Llama-3.3-70B-Instruct', allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])"
step config; $PY - $D/theirs/sweep.pt $EXPS <<'P'
import sys, torch
from huggingface_hub import hf_hub_download
cfg = torch.load(hf_hub_download("lu-christina/assistant-axis-vectors", "llama-3.3-70b/capping_config.pt", repo_type="dataset"), weights_only=False)
exps = [e for e in cfg["experiments"] if e["id"] in sys.argv[2:]]
assert len(exps) == len(sys.argv[2:]), [e["id"] for e in exps]
torch.save({"vectors": cfg["vectors"], "experiments": exps}, sys.argv[1])
print("config", [e["id"] for e in exps])
P
step sweep; (cd $CS && env -u PYTHONPATH uv run python /workspace/lu_code/1_capping_vllm.py --config_filepath $D/theirs/sweep.pt \
  --prompts_file /workspace/repo/wild/data/lu/theirs_1075.jsonl --output_jsonl $D/theirs/sweep_llama.jsonl \
  --model_name unsloth/Llama-3.3-70B-Instruct --max_model_len 4096 --company Meta --tensor_parallel_size 2 \
  --experiment_ids $EXPS) || echo "FAILED sweep"
step done
