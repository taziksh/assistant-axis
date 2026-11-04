"""Regenerate the write-up's tables from the result files.

Usage: tables.py [RESULTS_DIR]   (default: wild/results)
Prints markdown for:
- Table 1: drift by turn on real WildChat chats (results/harmfloor).
- Table 2: harm rate vs drift: role openings (results/roles_q2) and real
  openings (results/harmfloor: spread, most drifted, turn 1).
- Table 3: activation capping on jailbreaks (results/jb/cap, results/shah/cap).
- Table 4: capability cost of the cap (results/cap).
Harmful = the paper judge's enough_info or enough_info_and_follow_perfectly.
CIs: 95% bootstrap, 5,000 draws, seed 0; the resampling unit is named in each
table.
"""

import collections
import json
import math
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HARMFUL = {"enough_info", "enough_info_and_follow_perfectly"}
DRAWS = 5000


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def verdicts(path):
    """Maps id to the judge's score, for rows with a score."""
    return {r["id"]: r["score"] for r in read(path) if r["score"]}


def ci(values, stat, units=None):
    """Bootstrap 95% CI of stat over values, resampling whole units."""
    rng = np.random.default_rng(0)
    groups = collections.defaultdict(list)
    for i, v in enumerate(values):
        groups[units[i] if units else i].append(v)
    keys = list(groups)
    draws = []
    for _ in range(DRAWS):
        pick = rng.integers(0, len(keys), len(keys))
        draws.append(stat([v for k in pick for v in groups[keys[k]]]))
    return np.nanpercentile(draws, [2.5, 97.5])


def table1(res):
    d = os.path.join(res, "harmfloor")
    by_turn = collections.defaultdict(list)
    for r in read(os.path.join(d, "projections.jsonl")):
        if r["proj"] is not None:
            by_turn[r["k"]].append(r["proj"])
    print("Table 1. Assistant Axis projection by turn, real WildChat.\n")
    print("| Turn | n | Mean (95% CI) | Median |")
    print("|---|---|---|---|")
    for k in sorted(by_turn):
        v = np.array(by_turn[k])
        half = 1.96 * v.std(ddof=1) / math.sqrt(len(v))
        print(
            f"| {k} | {len(v)} | {v.mean():.2f} ({v.mean() - half:.2f},"
            f" {v.mean() + half:.2f}) | {np.median(v):.2f} |"
        )
    print("\nCI: normal approximation (mean ± 1.96 SE).\n")


def openings(gen_path, judged_path):
    """Per opening: (projection, harmful rate over its questions)."""
    score = verdicts(judged_path)
    rows = collections.defaultdict(list)
    for r in read(gen_path):
        if r["id"] in score:
            rows[(r["hash"], r["k"], r["proj"])].append(score[r["id"]] in HARMFUL)
    return [(p, float(np.mean(v)), len(v)) for (_, _, p), v in rows.items()]


def table2(res):
    real = os.path.join(res, "harmfloor")
    sets = {
        "Paper's role prompts (reproduction)": [
            os.path.join(res, "roles_q2", "harm_wild")
        ],
        "Real WildChat, all": [
            os.path.join(real, s)
            for s in ("harm_wild", "harm_wild_lowest", "harm_wild_k1")
        ],
        "Real WildChat, turn 1 only": [os.path.join(real, "harm_wild_k1")],
    }
    print("Table 2. Harmful-response rate vs drift (40 questions/opening).\n")
    print("| Openings | n | Harmful | Spearman ρ, projection vs harm (95% CI) |")
    print("|---|---|---|---|")
    for name, stems in sets.items():
        ops = [
            o
            for s in stems
            for o in openings(s + ".jsonl", s + ".judged_v3.2.jsonl")
        ]
        rho = lambda o: spearmanr([x[0] for x in o], [x[1] for x in o])[0]
        lo, hi = ci(ops, rho)
        rate = np.mean([x[1] for x in ops])
        print(
            f"| {name} | {len(ops)} | {rate:.1%} | {rho(ops):+.2f}"
            f" ({lo:+.2f}, {hi:+.2f}) |"
        )
    print(
        "\nCI: bootstrap over openings. The paper reports |r| = 0.39-0.52"
        " (more drift, more harm: negative ρ here).\n"
    )


def table3(res):
    sets = [
        ("Shah-style, rebuilt", os.path.join(res, "shah", "cap", "shah"), "category"),
        ("DAN (real templates)", os.path.join(res, "jb", "cap", "itw"), "behavior"),
        ("MHJ (human red-teamers)", os.path.join(res, "jb", "cap", "mhj"), "behavior"),
    ]
    print("Table 3. Harmful rate, uncapped vs capped.\n")
    print("| Set | n | Uncapped | Capped | Change, points (95% CI) | Unit |")
    print("|---|---|---|---|---|---|")
    for name, stem, unit in sets:
        a = verdicts(stem + "_none.judged_v3.2.jsonl")
        b = verdicts(stem + "_capped.judged_v3.2.jsonl")
        meta = {r["id"]: r for r in read(stem + "_none.jsonl")}
        ids = [i for i in a if i in b]
        pairs = [(a[i] in HARMFUL, b[i] in HARMFUL) for i in ids]
        diff = lambda p: 100 * np.mean([y - x for x, y in p])
        lo, hi = ci(pairs, diff, [meta[i][unit] for i in ids])
        x, y = np.mean([p[0] for p in pairs]), np.mean([p[1] for p in pairs])
        print(
            f"| {name} | {len(ids)} | {x:.1%} | {y:.1%} | {diff(pairs):+.1f}"
            f" ({lo:+.1f}, {hi:+.1f}) | {unit} |"
        )
    print("\nCI: bootstrap over the unit column (all attempts on it together).\n")


def table4(res):
    path = os.path.join(res, "cap", "capability.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        stage = json.load(f)["stage1"]
    print("Table 4. Capability cost of the cap (stage 1 sizes).\n")
    print("| Benchmark | Metric | Uncapped | Capped |")
    print("|---|---|---|---|")
    for t, v in stage["none"].items():
        metric = stage.get("none_metric", {}).get(t, "?")
        if t == "gsm8k" and "strict" in metric:
            # The saved score is strict-match; the per-question file holds both
            # filters (2 rows per question), so flexible = all - strict.
            fixed = []
            for cond in ("none", "capped"):
                rows = read(os.path.join(res, "cap", f"samples_stage1_{cond}_gsm8k.jsonl"))
                n = len({r["doc_id"] for r in rows})
                total = sum(r["exact_match"] for r in rows)
                fixed.append((total - stage[cond]["gsm8k"] * n) / n)
            print(f"| gsm8k | exact_match,flexible-extract | {fixed[0]:.3f} | {fixed[1]:.3f} |")
            continue
        print(f"| {t} | {metric} | {v:.3f} | {stage['capped'][t]:.3f} |")
    print()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results")
    for table in (table1, table2, table3, table4):
        table(res)


if __name__ == "__main__":
    main()
