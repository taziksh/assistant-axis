#!/usr/bin/env bash
# Laptop side of go_capfire.sh: collect its inputs from the saved runs. Usage: bash capfire_inputs.sh  (from wild/)
# Writes results/capfire_inputs/<model>/{real,filler,nohist}/{selected,drift,projections}.jsonl and
# jb/{pairs,shah_none}.jsonl, and one results/capfire_inputs/capfire_inputs_<model>.tgz per model. On the pod:
#   mkdir -p /workspace/wild/capfire && tar xzf capfire_inputs_<model>.tgz -C /workspace/wild/capfire
# jb holds the first 150 of Lu et al.'s jailbreak prompts and our uncapped replies to them (capdiag.py's 150).
# Those prompts are kept private (data/lu/PROVENANCE.md): the archive is for our own pod only.
set -euo pipefail
cd "$(dirname "$0")"
O=results/capfire_inputs
put() {  # MODEL CONDITION SOURCE_DIR
  mkdir -p $O/$1/$2
  cp $3/selected.jsonl $3/drift.jsonl $3/projections.jsonl $O/$1/$2/
}
put llama real results/drift_turns
put llama filler results/lc_llama/filler
put llama nohist results/lc_llama/nohist
put qwen real results/lc_qwen/real
put qwen filler results/lc_qwen/filler
put qwen nohist results/lc_qwen/nohist
jb() {  # MODEL PAIRS REPLIES
  mkdir -p $O/$1/jb
  head -150 $2 > $O/$1/jb/pairs.jsonl
  python3 - $O/$1/jb/pairs.jsonl $3 $O/$1/jb/shah_none.jsonl <<'P'
import json, sys
ids = {json.loads(l)["id"] for l in open(sys.argv[1])}
rows = [l for l in open(sys.argv[2]) if json.loads(l)["id"] in ids]
assert len(rows) == len(ids) == 150, (len(rows), len(ids))
open(sys.argv[3], "w").writelines(rows)
P
}
jb llama data/lu/pairs.jsonl results/lu/pod2/cap/shah_none.jsonl
jb qwen data/lu/pairs_qwen.jsonl results/lu/pod2/qwen/cap/shah_none.jsonl
for m in llama qwen; do
  tar czf $O/capfire_inputs_$m.tgz -C $O $m
  echo "$m: $(du -h $O/capfire_inputs_$m.tgz | cut -f1) $O/capfire_inputs_$m.tgz"
done
