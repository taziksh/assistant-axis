#!/usr/bin/env bash
# Paper 4.3 (harm vs drift) on Qwen 3 32B (user's go 2026-11-11): the Llama run (go_harm_floor.sh +
# go_roles.sh) with AXIS_MODEL=qwen (Qwen 3 32B, released Qwen axis at layer 32, thinking off). Same openings
# (the Llama runs' selected.jsonl files), same CatQA questions, same harm_wild.py stages and settings.
# Judging runs on the laptop. Stops the pod when done (after up to 60 min for the laptop to fetch); a backstop
# stops it 5 h after launch (weights on the volume: /dev/shm is too small for Qwen on a 1-GPU pod) regardless.
set -uo pipefail
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
export AXIS_MODEL=qwen HF_HOME=/workspace/hf PYTHONPATH=/workspace/repo PATH=/workspace/venv/bin:/root/.local/bin:$PATH
PY=/workspace/venv/bin/python
Q=data/dl/chqa/data/catqa_english.json
RR=/workspace/wild/qroles; RF=/workspace/wild/qfloor
cd /workspace/repo/wild
exec >> /root/qharm.log 2>&1  # local disk: appends to the network volume were being lost
step() { echo "== $(date -u +%FT%TZ) $*"; }
STOP="/usr/bin/runpodctl stop pod $RUNPOD_POD_ID"
step "stop command check"; /usr/bin/runpodctl get pod $RUNPOD_POD_ID || echo "WARNING: runpodctl cannot see this pod; laptop stop is the fallback"
setsid nohup bash -c "sleep 18000; $STOP" > /dev/null 2>&1 < /dev/null &
trap 'echo "== $(date -u +%FT%TZ) end"; cp /root/qharm.log /workspace/wild/qharm.log; touch /workspace/.qharm_done; for i in $(seq 180); do [ -f /workspace/.qharm_fetched ] && break; sleep 20; done; '"$STOP" EXIT
[ -x $PY ] && $PY -c "import vllm" 2>/dev/null || { step setup; (cd /workspace/repo && bash sanity_check/scripts/setup.sh); }
step qwen weights; $PY -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-32B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])"
$PY -c "import wcommon as w; print('config', w.MODEL, w.LAYER, w.AXIS_FILE, w.CAPPING_FILE, w.CAPPING_EXPERIMENT, w.CHAT_KW)"
# Role openings (paper 4.3): the Llama run's 550.
[ -f $RR/drift.jsonl ] || { step roles pool; $PY harm_wild.py $RR $Q pool || echo "FAILED roles pool"; }
step roles activations; $PY activations.py $RR || echo "FAILED roles activations"
[ -f $RR/harm_wild.jsonl ] || { step roles harm; $PY harm_wild.py $RR $Q harm --per-band 100000 --per-category 4 || echo "FAILED roles harm"; }
# Real WildChat openings: the Llama run's 2,000 random conversations.
[ -f $RF/drift.jsonl ] || { step wild pool; $PY harm_wild.py $RF $Q pool || echo "FAILED wild pool"; }
step wild activations; $PY activations.py $RF || echo "FAILED wild activations"
[ -f $RF/harm_wild.jsonl ] || { step wild harm bands; $PY harm_wild.py $RF $Q harm --per-band 6 --per-category 4 || echo "FAILED wild harm"; }
[ -f $RF/harm_wild_lowest.jsonl ] || { step wild harm lowest; $PY harm_wild.py $RF $Q harm --lowest 150 --per-category 4 --tag _lowest || echo "FAILED wild lowest"; }
[ -f $RF/harm_wild_k1.jsonl ] || { step wild harm k1; $PY harm_wild.py $RF $Q harm --only-k 1 --lowest 100000 --per-category 4 --tag _k1 || echo "FAILED wild k1"; }
step done
