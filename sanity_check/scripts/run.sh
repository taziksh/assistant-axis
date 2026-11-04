#!/usr/bin/env bash
# Run the sanity check end to end; finished steps are skipped on rerun. In tmux, from this dir:
#   NAME=pilot PER_DOMAIN=4 STOP_POD=0 bash run.sh
#   NAME=full PER_DOMAIN=50 bash run.sh   # stops the pod on exit (done or failed) or after MAX_HOURS
set -euo pipefail
cd "$(dirname "$0")"
: "${NAME:?}" "${PER_DOMAIN:?}"
export HF_HOME=${HF_HOME:-/workspace/hf}
PY=${PY:-/workspace/venv/bin/python}
DATA=${DATA:-/workspace/sanity}
RUN=$DATA/$NAME
mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1

stop_pod() { echo "== stopping pod"; runpodctl stop pod "$RUNPOD_POD_ID"; }
if [ "${STOP_POD:-1}" = 1 ]; then
  command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ] || { echo "runpodctl or RUNPOD_POD_ID missing"; exit 1; }
  trap stop_pod EXIT
  (sleep $(( ${MAX_HOURS:-10} * 3600 )); stop_pod) &
fi
step() { echo "== $(date -u +%FT%TZ) $*"; }
export PATH="$(dirname "$PY"):$PATH"  # FlashInfer JIT needs the venv's ninja.

# The network volume reads at ~0.4 GB/s, so each model load took ~8 min; load from RAM instead.
# /dev/shm is lost when the pod stops; the copy is redone then.
SHM=/dev/shm/hf
need_kb=$(du -sk "$HF_HOME/hub" | cut -f1); free_kb=$(df -k --output=avail /dev/shm | tail -1)
if [ ! -d "$SHM/hub" ] && [ "$free_kb" -lt $(( need_kb + 20000000 )) ]; then
  echo "== /dev/shm too small ($((free_kb / 1048576)) GiB free); loading weights from $HF_HOME"
  SHM=$HF_HOME
elif [ ! -d "$SHM/hub" ]; then
  step "copy weights to $SHM"
  rm -rf "$SHM.tmp" && mkdir -p "$SHM.tmp/hub"
  (cd "$HF_HOME/hub" && find . -type d -exec mkdir -p "$SHM.tmp/hub/{}" \; \
     && find . ! -type d -print0 | xargs -0 -P8 -I{} cp -a {} "$SHM.tmp/hub/{}")
  mv "$SHM.tmp" "$SHM"
fi
export HF_HOME=$SHM

[ -f "$DATA/convs.jsonl" ] || { step build; $PY build.py "$DATA/convs.jsonl"; }
[ -f "$RUN/selected.jsonl" ] || { step judge; $PY judge.py "$DATA/convs.jsonl" "$DATA/labels.jsonl" "$RUN/selected.jsonl" --per-domain "$PER_DOMAIN"; }
step generate; $PY generate.py "$RUN/selected.jsonl" "$RUN"  # Only generates writers not yet in scored.jsonl.
[ -f "$RUN/turn_labels.jsonl" ] || { step label; $PY label_turns.py "$RUN"; }
step activations; $PY activations.py "$RUN"
step analyze; $PY analyze.py "$RUN"
step done
