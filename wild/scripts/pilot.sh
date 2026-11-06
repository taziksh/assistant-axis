#!/usr/bin/env bash
# Run one pilot stage in the existing pod environment. See wild/docs/runpod.md.
set -euo pipefail
cd "$(dirname "$0")/../.."

stage=${1:-}
case "$stage" in
  smoke|validate|select|measure) ;;
  *) echo 'Usage: bash wild/scripts/pilot.sh smoke|validate|select|measure' >&2; exit 2 ;;
esac
if [ "$#" -ne 1 ]; then
  echo 'Expected exactly one stage: smoke, validate, select, or measure.' >&2
  exit 2
fi

export HF_HOME=/workspace/palaestra/hf
export HF_HUB_CACHE="$HF_HOME/hub"
# The prepared pod exports 1 but does not have the optional hf_transfer package.
export HF_HUB_ENABLE_HF_TRANSFER=0
unset TRANSFORMERS_CACHE
export PYTHONUNBUFFERED=1
export PATH="$PWD/.venv/bin:$PATH"
py="$PWD/.venv/bin/python"
model_revision=99cd0d2c829e92a67c844f9144c2509632e5c87f
pool=${CANDIDATE_POOL:-wild/data/drift_candidates_v3.jsonl}
smoke_dir=${SMOKE_DIR:-wild/results/drift_smoke_v3}
run_dir=${RUN_DIR:-wild/results/drift_v3}
references=${CLASSIFIER_REFERENCES:-wild/results/selection_validation_v3/references.jsonl}
validation_dir=${VALIDATION_DIR:-wild/results/selection_validation_v3/model_run}
per_domain=${PER_DOMAIN:-100}
if [ "$stage" = select ] && ! [[ "$per_domain" =~ ^[1-9][0-9]*$ ]]; then
  echo 'PER_DOMAIN must be a positive integer.' >&2
  exit 2
fi

[ -x "$py" ] || { echo "Missing environment: $py" >&2; exit 1; }
[ -f "$pool" ] && [ -f "$pool.manifest.json" ] || { echo 'Candidate pool or provenance sidecar is missing.' >&2; exit 1; }
snapshot="$HF_HUB_CACHE/models--unsloth--Llama-3.3-70B-Instruct/snapshots/$model_revision"
[ -d "$snapshot" ] || { echo "Existing model cache is missing: $snapshot" >&2; exit 1; }

# Readiness is a completed, fingerprinted diagnostic result, never a log message.
smoke_axis() {
  "$py" - "$smoke_dir" "$model_revision" "$pool" <<'PY'
from pathlib import Path
import sys
from wild.drift.smoke import load_run
from wild.drift.state import checksum, read_record

root = Path(sys.argv[1])
run = load_run(root)
result = read_record(root / "smoke.result.json")
if result.get("status") != "complete":
    raise SystemExit("Smoke has not completed successfully; inspect its log/result.")
if result["run_id"] != run["run_id"]:
    raise SystemExit("Smoke result does not match its manifest.")
pool = Path(sys.argv[3])
if (checksum(pool.read_bytes()) != run["source"]["candidates_sha256"]
        or checksum(Path(str(pool) + ".manifest.json").read_bytes()) != run["source"]["candidate_manifest_sha256"]):
    raise SystemExit("Candidate pool or provenance changed since smoke; run a fresh smoke first.")
if result["artifacts"]["model"]["revision"] != sys.argv[2]:
    raise SystemExit("Smoke used a different model revision.")
for name, expected in result["outputs"].items():
    if checksum((root / name).read_bytes()) != expected:
        raise SystemExit(f"Smoke output changed: {name}")
print(result["artifacts"]["axis"]["revision"])
PY
}

run_stage() {
  case "$stage" in
    smoke)
      "$py" -m wild.drift.smoke "$pool" "$smoke_dir" --model-revision "$model_revision"
      smoke_axis >/dev/null
      echo "Smoke passed. Inspect $smoke_dir/smoke.result.json before running validate."
      ;;
    validate)
      smoke_axis >/dev/null
      "$py" -m wild.selection.evaluate "$references" "$validation_dir" --model-revision "$model_revision" --max-turns 15
      echo "Inspect $validation_dir/metrics.json and disagreements.jsonl before full selection; agreement with provisional references is not human validation."
      ;;
    select)
      smoke_axis >/dev/null
      "$py" -m wild.selection.judge "$pool" "$run_dir" --per-domain "$per_domain" --min-turns 3 --max-turns 15 --model-revision "$model_revision"
      "$py" - "$run_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = [json.loads(line) for line in (root / "selected.jsonl").read_bytes().split(b"\n") if line.strip()]
labels = {row["hash"]: row for row in (json.loads(line) for line in (root / "labels.jsonl").read_bytes().split(b"\n") if line.strip())}
with (root / "selection.review.txt").open("w") as out:
    for row in rows:
        out.write(f"\n{'=' * 72}\n{row['hash']} | {row['domain']} | {row['subtopic']}\n{labels[row['hash']]['rationale']}\n")
        for k, (user, assistant) in enumerate(zip(row["users"], row["gpt"]), 1):
            out.write(f"\nUSER {k}: {user}\nRECORDED ASSISTANT {k}: {assistant}\n")
print(f"Review {root / 'selection.review.txt'} and the printed category counts before running measure.")
PY
      ;;
    measure)
      axis_revision=$(smoke_axis)
      "$py" - "$run_dir" "$pool" <<'PY'
import json
from pathlib import Path
import sys
from wild.selection import helpers

root = Path(sys.argv[1])
manifest = json.loads((root / "selected.manifest.json").read_text())
if helpers.file_checksum(root / "selected.jsonl") != manifest["selected_sha256"]:
    raise SystemExit("Selected records changed since selection.")
pool = Path(sys.argv[2])
for filename, field in (("labels.jsonl", "labels_sha256"), ("selection.audit.jsonl", "audit_sha256")):
    if helpers.file_checksum(root / filename) != manifest[field]:
        raise SystemExit(f"Frozen selection artifact changed: {filename}")
quota = helpers.verify_selection_quotas(
    manifest, helpers.read_jsonl(root / "selected.jsonl"), helpers.read_jsonl(pool),
    helpers.read_jsonl(root / "labels.jsonl"), helpers.read_jsonl(root / "selection.audit.jsonl"))
if (manifest["identity"]["candidates_sha256"] != helpers.file_checksum(pool)
        or manifest["identity"]["candidate_manifest_sha256"] != helpers.file_checksum(Path(str(pool) + ".manifest.json"))):
    raise SystemExit("Selection and successful smoke must use the same candidate pool and provenance.")
print(f"Measurement uses the frozen selection quota: {quota} per category, best-effort subtopic balance, max {manifest['selection']['max_turns']} turns.")
PY
      "$py" -m wild.drift "$run_dir" --samples-per-turn 3 --model-revision "$model_revision" --axis-revision "$axis_revision"
      echo "Measurement and analysis completed: $run_dir/analysis/REPORT.md"
      ;;
  esac
}

mkdir -p wild/results/logs
log="wild/results/logs/pilot-$stage-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
echo "Running $stage; live log: $log"
run_stage 2>&1 | tee "$log"
