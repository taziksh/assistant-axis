#!/usr/bin/env bash
# Drift on real coding-agent sessions (SWE-chat), user's go 2026-11-06. Same steps as the WildChat drift run
# (drift_turns): Llama's reply at every user turn 1-12 after the real prompts and the agent's original text
# replies, then layer-40 projections. Needs /workspace/wild/swechat/selected.jsonl (swechat_build.py, run on the
# laptop because SWE-chat is gated). The pod stops itself when done (after up to 30 min for the laptop to fetch)
# or on any failure; a separate backstop stops it after 6 h regardless.
set -uo pipefail
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
export HF_HOME=/dev/shm/hf PYTHONPATH=/workspace/repo PATH=/workspace/venv/bin:$PATH
PY=/workspace/venv/bin/python
D=/workspace/wild/swechat; Q=data/dl/chqa/data/catqa_english.json
cd /workspace/repo/wild
exec >> $D/run.log 2>&1
trap 'echo "== $(date -u +%FT%TZ) end"; touch /workspace/.swechat_done; for i in $(seq 90); do [ -f /workspace/.swechat_fetched ] && break; sleep 20; done; /usr/bin/runpodctl stop pod "$RUNPOD_POD_ID"' EXIT
step() { echo "== $(date -u +%FT%TZ) $*"; }
[ -x $PY ] && $PY -c "import vllm" 2>/dev/null || { step setup; (cd /workspace/repo && bash sanity_check/scripts/setup.sh); }
step weights; $PY -c "from huggingface_hub import snapshot_download as s; s('unsloth/Llama-3.3-70B-Instruct', allow_patterns=['*.json', '*.safetensors', 'tokenizer*'])"
[ -f $D/drift.jsonl ] || { step pool; $PY harm_wild.py $D $Q pool --all-turns; }
step activations; $PY activations.py $D
step done
