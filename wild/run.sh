#!/usr/bin/env bash
# Drift, harm and capping on real chats, end to end; finished steps are skipped on rerun.
# In tmux, from this dir, with data/dl/ copied over:
#   NAME=main PER_DOMAIN=150 bash run.sh
#   JAILBREAK_ONLY=1 NAME=jb PER_DOMAIN=0 bash run.sh   # only the jailbreak/capping/capability steps
# Stops the pod on exit (done or failed) or after MAX_HOURS.
set -euo pipefail
cd "$(dirname "$0")"
: "${NAME:?}" "${PER_DOMAIN:?}"
export HF_HOME=${HF_HOME:-/workspace/hf}
PY=${PY:-/workspace/venv/bin/python}
RUN=/workspace/wild/$NAME
DL=$PWD/data/dl
mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
export PYTHONPATH="$PWD/..:${PYTHONPATH:-}"

stop_pod() { echo "== stopping pod"; runpodctl stop pod "$RUNPOD_POD_ID"; }
if [ "${STOP_POD:-1}" = 1 ]; then
  command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ] || { echo "runpodctl or RUNPOD_POD_ID missing"; exit 1; }
  trap stop_pod EXIT
  (sleep $(( ${MAX_HOURS:-12} * 3600 )); stop_pod) &
fi
step() { echo "== $(date -u +%FT%TZ) $*"; }
export PATH="$(dirname "$PY"):$PATH"  # FlashInfer JIT needs the venv's ninja.

step setup
command -v uv >/dev/null || pip install -q --break-system-packages uv  # The container disk resets on stop.
# transformers and vllm stay at the versions setup.sh installed for the history-swap check.
uv pip install -q --python "$PY" pandas matplotlib "sentence-transformers>=3,<6" "lm_eval[ifeval]==0.4.9.2" \
  transformers==4.57.6 vllm==0.23.0
$PY -c "
from huggingface_hub import hf_hub_download as d, snapshot_download as s
s('unsloth/Llama-3.3-70B-Instruct', allow_patterns=['*.json', '*.safetensors', 'tokenizer*'])  # RAM copy is lost on stop.
s('cais/HarmBench-Llama-2-13b-cls'); s('Qwen/Qwen3-Embedding-0.6B')
for f in ['llama-3.3-70b/assistant_axis.pt', 'llama-3.3-70b/capping_config.pt']:
    d('lu-christina/assistant-axis-vectors', f, repo_type='dataset')"

# Load weights from RAM when HF_HOME is on a network volume (~0.4 GB/s); SKIP_SHM=1 for local disk.
# /dev/shm is lost when the pod stops.
SHM=/dev/shm/hf
if [ "${SKIP_SHM:-0}" = 1 ]; then SHM=$HF_HOME; fi
need_kb=$(du -sk "$HF_HOME/hub" | cut -f1); free_kb=$(df -k --output=avail /dev/shm | tail -1)
if [ "$SHM" != "$HF_HOME" ] && [ -d "$SHM/hub" ] && [ "$(du -sk "$SHM/hub" | cut -f1)" -lt "$need_kb" ]; then rm -rf "$SHM"; fi  # Stale copy.
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

if [ "${CAPABILITY_ONLY:-0}" = 1 ]; then
  step capability; $PY capability.py "$RUN"
  step done
  exit 0
fi
if [ "${JAILBREAK_ONLY:-0}" = 1 ]; then
  # Jailbreak harm (MHJ, uncapped first) + capping + capability; no WildChat steps. Harm is judged locally
  # with the paper's judge (judge_paper.py), so the HarmBench classifier is not run here.
  step capping; $PY capping.py "$RUN" "$DL"
  step capability; $PY capability.py "$RUN"
  step done
  exit 0
fi
[ -f data/convs.jsonl ] || { step build; $PY build.py data/convs.jsonl; }
[ -f "$RUN/selected.jsonl" ] || { step judge; $PY judge.py data/convs.jsonl "$RUN" --per-domain "$PER_DOMAIN"; }
[ -f "$RUN/drift.jsonl" ] || { step generate; $PY generate.py "$RUN" "$DL/harmbench_behaviors_text_all.csv" --drift-only; }
# Harm on real multi-turn jailbreaks first: uncapped MHJ replies, their projections, the judge.
step capping_mhj; $PY capping.py "$RUN" "$DL" --only mhj:none
step judge_harm; $PY judge_harm.py "$RUN"
step analyze; $PY analyze.py "$RUN"
step capping; $PY capping.py "$RUN" "$DL"
step judge_harm; $PY judge_harm.py "$RUN"
step analyze; $PY analyze.py "$RUN"
step activations
if [ -f "$RUN/drift.manifest.json" ]; then
  $PY drift_activations.py "$RUN"
else
  $PY activations.py "$RUN"
fi
# step capability; $PY capability.py "$RUN"  # On hold until the benchmark set is approved.
step analyze; $PY analyze.py "$RUN"
step done
