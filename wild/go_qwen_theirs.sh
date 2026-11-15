#!/usr/bin/env bash
# Their capping script on Qwen again, with thinking off (2026-11-08). As published, the script ignores its
# --thinking flag, so go_lu2.sh's Qwen run had thinking on (every reply has <think>), unlike their saved
# Qwen replies. 1_capping_vllm_nothink.py differs only by enable_thinking=False in the chat template.
# Launched by the laptop after go_sweep.sh finishes. Stops the pod when done (after up to 30 min for the
# laptop to fetch); the 4:30 am Toronto backstop still applies.
set -uo pipefail
export RUNPOD_POD_ID="$(tr "\0" "\n" < /proc/1/environ | grep "^RUNPOD_POD_ID=" | cut -d= -f2-)"
export HF_HOME=/dev/shm/hf PATH=/workspace/venv/bin:/root/.local/bin:$PATH
D=/workspace/wild/lu; CS=/root/git/chatspace
exec >> $D/qwen2.log 2>&1
trap 'echo "== $(date -u +%FT%TZ) end"; touch /workspace/.qwen2_done; for i in $(seq 90); do [ -f /workspace/.qwen2_fetched ] && break; sleep 20; done; /usr/bin/runpodctl stop pod $RUNPOD_POD_ID' EXIT
echo "== $(date -u +%FT%TZ) qwen weights"
/workspace/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-32B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])"
echo "== $(date -u +%FT%TZ) qwen theirs, thinking off"
(cd $CS && env -u PYTHONPATH uv run python /workspace/lu_code/1_capping_vllm_nothink.py --config_filepath $D/theirs/qwen.pt \
  --prompts_file /workspace/repo/wild/data/lu/theirs_1075.jsonl --output_jsonl $D/theirs/qwen_nothink.jsonl \
  --model_name Qwen/Qwen3-32B --max_model_len 4096 --company Alibaba --name Qwen --thinking false \
  --tensor_parallel_size 1 --experiment_ids layers_46:54-p0.25 nobind) || echo "FAILED qwen theirs nothink"
echo "== $(date -u +%FT%TZ) done"
