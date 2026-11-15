"""Capability tables from a capability.py run: scores and capped - uncapped.

Usage: capability_score.py RUN_DIR [STAGE]   (STAGE default: stage1)
Reads RUN_DIR/samples_{STAGE}_{none,capped}_{task}.jsonl (per-question scores)
and prints the two tables in the write-up:
- score per benchmark, uncapped and capped;
- change, capped - uncapped, with a 95% CI from the paired differences: mean
  +/- 1.96 x SE of the per-question differences ("Adding Error Bars to Evals",
  Miller 2024). This reproduces the write-up's Llama intervals exactly; a
  bootstrap over questions gives nearly the same ones.
Metrics, as reported for Llama 3.3 70B (results/cap):
- IFEval: strict accuracy per instruction, averaged within each prompt and then
  over prompts (0.919 uncapped on Llama; the prompt-level score is 0.875).
- MMLU Pro: exact_match (custom-extract) over all questions.
- GSM8K: exact_match with lm-eval's flexible-extract filter. Sample files from
  before capability.py saved the filter name hold two unlabeled rows per
  question, strict-match then flexible-extract; the second is used.
- EQ-Bench: mean eqbench score.
"""

import collections
import json
import os
import sys

import numpy as np

TASKS = ["ifeval", "mmlu_pro", "gsm8k", "eq_bench"]
NAMES = {
    "ifeval": "IFEval (per instruction)",
    "mmlu_pro": "MMLU Pro",
    "gsm8k": "GSM8K",
    "eq_bench": "EQ-Bench",
}


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def scores(run_dir, stage, cond, task):
    """(subtask, doc_id) -> score for one benchmark and condition."""
    rows = read(os.path.join(run_dir, f"samples_{stage}_{cond}_{task}.jsonl"))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["task"], r["doc_id"])].append(r)
    out = {}
    for key, rs in by.items():
        if task == "ifeval":
            out[key] = float(np.mean(rs[0]["inst_level_strict_acc"]))
        elif task == "gsm8k":
            named = [r for r in rs if r.get("filter") == "flexible-extract"]
            out[key] = float((named[0] if named else rs[-1])["exact_match"])
        elif task == "mmlu_pro":
            out[key] = float(rs[0]["exact_match"])
        else:
            out[key] = float(rs[0]["eqbench"])
    return out


def main():
    run_dir = sys.argv[1]
    stage = sys.argv[2] if len(sys.argv) > 2 else "stage1"
    marker = os.path.join(run_dir, "capability_model.txt")
    if os.path.exists(marker):
        print("Model and cap:", open(marker).read().strip(), "\n")
    rows = []
    for t in TASKS:
        try:
            a, b = (scores(run_dir, stage, c, t) for c in ("none", "capped"))
        except FileNotFoundError:
            continue
        keys = sorted(set(a) & set(b))
        if len(keys) != len(a) or len(keys) != len(b):
            print(f"WARNING {t}: {len(a)} uncapped and {len(b)} capped questions, {len(keys)} in both")
        x = np.array([a[k] for k in keys])
        y = np.array([b[k] for k in keys])
        d = y - x
        half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        rows.append(
            (t, len(keys), x.mean(), y.mean(), d.mean(), d.mean() - half, d.mean() + half)
        )
    print("| Benchmark | n | Uncapped | Capped |")
    print("|---|---|---|---|")
    for t, n, x, y, *_ in rows:
        f = ".2f" if t == "eq_bench" else ".3f"
        print(f"| {NAMES[t]} | {n} | {x:{f}} | {y:{f}} |")
    print("\n| Benchmark | Change, capped - uncapped (95% CI) |")
    print("|---|---|")
    for t, _, _, _, d, lo, hi in rows:
        f = "+.2f" if t == "eq_bench" else "+.3f"
        print(f"| {NAMES[t]} | {d:{f}} ({lo:{f}}, {hi:{f}}) |")


if __name__ == "__main__":
    main()
