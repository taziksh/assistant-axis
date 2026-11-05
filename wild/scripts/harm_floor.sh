#!/usr/bin/env bash
# Harm (paper 4.3) floor check on real WildChat openings: is Llama's harm rate after real chats measurable
# (not ~0 everywhere) with CategoricalHarmfulQA questions? Approved 2026-11-03 (60 openings x 40 questions).
set -euo pipefail
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export HF_HOME=/dev/shm/hf PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PATH=/workspace/venv/bin:$PATH
PY=/workspace/venv/bin/python
R=/workspace/wild/harmfloor; mkdir -p $R
cd "$REPO_DIR/wild"
exec >> $R/run.log 2>&1
trap 'echo "== stopping pod"; runpodctl stop pod "$RUNPOD_POD_ID"' EXIT
step() { echo "== $(date -u +%FT%TZ) $*"; }
# Fresh pod: venv and packages (same versions as before), weights go to RAM (HF_HOME above).
[ -x $PY ] && $PY -c "import vllm" 2>/dev/null || { step setup; (cd "$REPO_DIR" && bash sanity_check/scripts/setup.sh); }
$PY -c "from huggingface_hub import snapshot_download as s; s('unsloth/Llama-3.3-70B-Instruct', allow_patterns=['*.json', '*.safetensors', 'tokenizer*'])"
[ -f $R/selected.jsonl ] || { step build; $PY -m wild.selection.build $R/selected.jsonl --random 2000; }
[ -f $R/drift.jsonl ] || { step pool; $PY -m wild.harm.generate $R data/dl/chqa/data/catqa_english.json pool; }
step activations; $PY -m wild.legacy.measurement $R
[ -f $R/harm_wild.jsonl ] || { step harm; $PY -m wild.harm.generate $R data/dl/chqa/data/catqa_english.json harm --per-band 6 --per-category 4; }
step done
