#!/usr/bin/env bash
# Paper 4.3 on its own settings (user's go 2026-11-03): the 550 role openings (roles_build.py --questions 0 2),
# opening reply capped at 512 tokens (repo default), then 42 Shah-style questions (1 per category), temp 0.7 /
# top-p 0.9, 512 tokens. Then Qwen3Guard-8B on ToxicChat (classifier comparison). Judging runs on the laptop.
set -uo pipefail
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
export HF_HOME=/dev/shm/hf PYTHONPATH=/workspace/repo PATH=/workspace/venv/bin:$PATH
PY=/workspace/venv/bin/python
R=/workspace/wild/roles_shah; Q=data/dl/chqa/data/catqa_english.json
mkdir -p $R; cd /workspace/repo/wild
exec >> $R/run.log 2>&1
trap 'echo "== $(date -u +%FT%TZ) end"; touch /workspace/.shah_done; for i in $(seq 90); do [ -f /workspace/.shah_fetched ] && break; sleep 20; done; runpodctl stop pod "$RUNPOD_POD_ID"' EXIT
step() { echo "== $(date -u +%FT%TZ) $*"; }
[ -x $PY ] && $PY -c "import vllm" 2>/dev/null || { step setup; (cd /workspace/repo && bash sanity_check/scripts/setup.sh); }
step weights; $PY -c "from huggingface_hub import snapshot_download as s; s('unsloth/Llama-3.3-70B-Instruct', allow_patterns=['*.json', '*.safetensors', 'tokenizer*'])"
[ -f $R/selected.jsonl ] || { step build; $PY roles_build.py $R --questions 0 2; }
[ -f $R/drift.jsonl ] || { step pool; $PY harm_wild.py $R $Q pool --first-max-tokens 512; }
step activations; $PY activations.py $R
[ -f $R/harm_wild.jsonl ] || { step harm; $PY harm_wild.py $R $Q harm --per-band 100000 --questions-jsonl data/shah/questions_42.jsonl; }
[ -f data/cls/toxicchat_test.cls_qwen3guard.jsonl ] || { step qwen3guard; $PY classify.py data/cls/toxicchat_test.jsonl qwen3guard --batch 64; }
step done
