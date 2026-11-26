#!/usr/bin/env bash
# Harm after drift on real WildChat openings with the paper's own questions (Qwen 3 32B; user's go 2026-11-24,
# up to $20, Palaestra Modal workspace). Runs on the dev server in tmux, so no laptop is needed: starts the Modal
# run (modal_qreal.py: 365 openings x 20 questions, one H200, 2 h limit), waits for it, fetches the replies, judges
# them with the paper's judge (DeepSeek V3.2, Lu et al.'s judge text, as for results/q43b) and writes
# results/qreal20/summary.txt. A rerun resumes: saved replies and verdicts are kept.
set -uo pipefail
cd "$(dirname "$0")"
export MODAL_PROFILE=palaestra
set -a; . ../.env; set +a
PY=../.venv/bin/python; MODAL=../.venv/bin/modal
RUN=${1:-qreal20}  # qreal20 (20 questions) or qreal86 (all 86; resumes from qreal20's replies and verdicts)
R=results/$RUN
exec >> $R/go_qreal.log 2>&1
step() { echo "== $(date +%FT%T%z) $*"; }
if [ ! -f $R/call_id.txt ]; then
  step deploy; $MODAL deploy modal_qreal.py || { echo "FAILED deploy"; exit 1; }
  step prepare; $MODAL run modal_qreal.py::prepare --run $RUN || { echo "FAILED prepare"; exit 1; }
  step launch; $MODAL run modal_qreal.py::launch --run $RUN | tee $R/launch.txt
  grep -o "fc-[A-Za-z0-9]*" $R/launch.txt | head -1 > $R/call_id.txt
fi
step "wait for $(cat $R/call_id.txt)"; $PY qreal_wait.py "$(cat $R/call_id.txt)" 3 || echo "WARNING: the Modal call did not finish cleanly; using the replies saved so far"
step fetch; $MODAL volume get --force aa-qreal $RUN/harm_wild.s0.jsonl $R/harm_wild.s0.jsonl || { echo "FAILED fetch"; exit 1; }
step merge; RUN=$RUN $PY - <<'PYEOF'
import json
import os
R = "results/" + os.environ["RUN"] + "/"
persona = {json.loads(l)["id"]: json.loads(l)["persona"] for l in open(R + "questions.jsonl")}
rows = [json.loads(l) for l in open(R + "harm_wild.s0.jsonl") if l.strip()]
with open(R + "harm_real.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({**r, "persona": persona[r["question_id"]]}) + "\n")
print(len(rows), "replies from", len({(r["hash"], r["k"]) for r in rows}), "openings")
PYEOF
for pass in 1 2; do step "judge pass $pass"; $PY judge_paper.py $R/harm_real.jsonl --judge v3.2 --scenario lu --concurrency 32 || echo "FAILED judge pass $pass"; done
step stats; $PY qreal_stats.py $R | tee $R/summary.txt
step done; touch $R/DONE
