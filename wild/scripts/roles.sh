#!/usr/bin/env bash
# Paper 4.3 reproduction on Llama, stage 1: 550 role openings (roles_build.py --questions 0 2); plus real
# openings: the 150 most drifted and all turn-1 ones (both not already in the first 50),
# same pipeline, questions and judge as the harm floor check. User's go 2026-11-03.
set -uo pipefail
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export HF_HOME=/dev/shm/hf PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PATH=/workspace/venv/bin:$PATH
export OPENROUTER_API_KEY="$(cat /dev/shm/or_key)"
PY=/workspace/venv/bin/python
R=/workspace/wild/roles_q2; W=/workspace/wild/harmfloor; Q=data/dl/chqa/data/catqa_english.json
mkdir -p $R; cd "$REPO_DIR/wild"
exec >> $R/run.log 2>&1
trap 'echo "== $(date -u +%FT%TZ) end"; touch /workspace/.roles_done; for i in $(seq 90); do [ -f /workspace/.roles_fetched ] && break; sleep 20; done; rm -f /dev/shm/or_key; /usr/bin/runpodctl stop pod "$RUNPOD_POD_ID"' EXIT
step() { echo "== $(date -u +%FT%TZ) $*"; }
[ -f $R/selected.jsonl ] || { step build; $PY -m wild.harm.roles $R --questions 0 2; }
[ -f $R/drift.jsonl ] || { step pool; $PY -m wild.harm.generate $R $Q pool; }
step activations; $PY -m wild.legacy.measurement $R
[ -f $R/harm_wild.jsonl ] || { step harm roles; $PY -m wild.harm.generate $R $Q harm --per-band 100000 --per-category 4; }
[ -f $W/harm_wild_lowest.jsonl ] || { step harm lowest; $PY -m wild.harm.generate $W $Q harm --lowest 150 --per-category 4 --tag _lowest; }
[ -f $W/harm_wild_k1.jsonl ] || { step harm k1; $PY -m wild.harm.generate $W $Q harm --only-k 1 --lowest 100000 --per-category 4 --tag _k1; }
step judge roles; timeout 150m $PY -m wild.harm.paper_judge $R/harm_wild.jsonl --judge v3.2 --concurrency 96
step judge real; timeout 60m $PY -m wild.harm.paper_judge $W/harm_wild_lowest.jsonl $W/harm_wild_k1.jsonl --judge v3.2 --concurrency 96
step done
