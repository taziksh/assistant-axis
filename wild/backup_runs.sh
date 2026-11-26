#!/usr/bin/env bash
# Waits for the qreal86 run, then adds today's run folders to the private results backup
# (Hugging Face dataset taziksh/assistant-axis-wild-results, under results/). Runs on dev in tmux.
cd "$(dirname "$0")"
exec >> results/backup_runs.log 2>&1
until [ -f results/qreal86/DONE ] || ! tmux has-session -t qreal86 2>/dev/null; do sleep 60; done
echo "== $(date +%FT%T%z) run finished (DONE: $([ -f results/qreal86/DONE ] && echo yes || echo no)); uploading"
../.venv/bin/python - <<"PY"
import time
from huggingface_hub import HfApi
api = HfApi()
for run in ("qreal20", "qreal86"):
    for attempt in range(6):
        try:
            api.upload_folder(folder_path=f"results/{run}", path_in_repo=f"results/{run}", repo_id="taziksh/assistant-axis-wild-results", repo_type="dataset", commit_message=f"Add {run} (harm after drift, real openings x the paper questions)")
            print(run, "uploaded", flush=True)
            break
        except Exception as e:
            print(run, "attempt", attempt + 1, "failed:", repr(e)[:300], flush=True)
            time.sleep(60)
PY
echo "== $(date +%FT%T%z) backup done"
