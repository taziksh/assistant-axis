"""Regenerate the write-up's tables from the result files.

Usage: tables.py [RESULTS_DIR]   (default: wild/results)
Prints markdown for:
- Table 1: drift by turn on real WildChat chats (results/harmfloor).
- Table 1b: drift by turn in the same chats (results/drift_turns).
- Table 1c: our projection of the paper's released Llama conversations vs its
  Fig. 25 (results/drift_scale).
- Table 2: harm rate vs drift: role openings (results/roles_q2) and real
  openings (results/harmfloor: spread, most drifted, turn 1).
- Table 3: activation capping on jailbreaks (results/jb/cap for MHJ,
  results/dan3/cap for DAN, results/shah/cap).
- Table 4: capability cost of the cap (results/cap).
- Appendix A1-A2: Table 3 by each source dataset's own categories (MHJ: HarmBench
  semantic categories; DAN: TrustAIRLab's 13). A3: Table 2's harmful rate by
  CategoricalHarmfulQA category and opening type.
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


def table1b(res):
    path = os.path.join(res, "drift_turns", "projections.jsonl")
    if not os.path.exists(path):
        return
    by_chat = collections.defaultdict(dict)
    for r in read(path):
        if r["proj"] is not None:
            by_chat[r["hash"]][r["k"]] = r["proj"]
    turns = range(1, 13)
    chats = [c for c in by_chat.values() if all(k in c for k in turns)]
    print(
        "Table 1b. Assistant Axis projection by turn, the same real WildChat"
        f" chats at every turn ({len(chats)} chats with all 12 turns).\n"
    )
    print("| Turn | Mean (95% CI) | Change from turn 1 (95% CI) |")
    print("|---|---|---|")
    for k in turns:
        v = [c[k] for c in chats]
        d = [c[k] - c[1] for c in chats]
        lo, hi = ci(v, np.mean)
        dlo, dhi = ci(d, np.mean)
        change = (
            "—" if k == 1 else f"{np.mean(d):+.2f} ({dlo:+.2f}, {dhi:+.2f})"
        )
        print(f"| {k} | {np.mean(v):.2f} ({lo:.2f}, {hi:.2f}) | {change} |")
    print("\nCI: bootstrap over chats.\n")


# The simulated user that wrote each released Llama transcript.
RELEASED_USER = {"coding": "Sonnet 4.5", "therapy": "GPT-5"}


def paper_fig25():
    """(domain, user) -> per-turn values digitized from the paper's Fig. 25."""
    with open(os.path.join(os.path.dirname(__file__), "paper_fig25.json")) as f:
        return {tuple(k.split("|")): v for k, v in json.load(f).items()}


def table1c(res):
    path = os.path.join(res, "drift_scale", "projections.jsonl")
    if not os.path.exists(path):
        return
    by_conv = collections.defaultdict(dict)
    for r in read(path):
        by_conv[r["hash"]][r["k"]] = r["proj"]
    print(
        "Table 1c. Scale check: our projection of the paper's released Llama"
        " conversations vs the paper's Fig. 25.\n"
    )
    print(
        "| Domain (simulated user) | Ours (turn 1) | Paper (turn 1) | Ours"
        " (last 3 turns) | Paper (last 3 turns) |"
    )
    print("|---|---|---|---|---|")
    paper = paper_fig25()
    for domain, user in RELEASED_USER.items():
        v = by_conv[f"paper-{domain}"]
        ks = sorted(v)
        ours_last = np.mean([v[k] for k in ks[-3:]])
        curve = [x for x in paper[(domain, user)] if x is not None]
        first, last = curve[0], np.mean(curve[-3:])
        print(
            f"| {domain} ({user}) | {v[ks[0]]:.2f} | {first:.2f} |"
            f" {ours_last:.2f} | {last:.2f} |"
        )
    print(
        "\nOurs: one released conversation per domain. Paper: average of 100"
        " conversations, digitized from the figure (paper_fig25.json).\n"
    )


def openings(gen_path, judged_path):
    """Per opening: (projection, harmful rate over its questions)."""
    score = verdicts(judged_path)
    rows = collections.defaultdict(list)
    for r in read(gen_path):
        if r["id"] in score:
            rows[(r["hash"], r["k"], r["proj"])].append(
                score[r["id"]] in HARMFUL
            )
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
    print(
        "| Openings | n | Harmful | Spearman ρ, projection vs harm (95% CI) |"
    )
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
        (
            "Shah et al. (regenerated)",
            os.path.join(res, "shah", "cap", "shah"),
            "category",
        ),
        (
            "DAN (3 templates per question)",
            os.path.join(res, "dan3", "cap", "itw"),
            "behavior",
        ),
        (
            "MHJ (human red-teamers)",
            os.path.join(res, "jb", "cap", "mhj"),
            "behavior",
        ),
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
    print(
        "\nCI: bootstrap over the unit column (all attempts on it together).\n"
    )


def capability_scores(res):
    """Benchmark -> (metric, uncapped, capped, SEs), or {} without results."""
    path = os.path.join(res, "cap", "capability.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        stage = json.load(f)["stage1"]
    out = {}
    for t, v in stage["none"].items():
        metric = stage.get("none_metric", {}).get(t, "?")
        scores = [v, stage["capped"][t]]
        ses = [stage["none_stderr"][t], stage["capped_stderr"][t]]
        if t == "gsm8k" and "strict" in metric:
            # The saved score is strict-match; the per-question file holds both
            # filters (2 rows per question), so flexible = all - strict.
            for c, cond in enumerate(("none", "capped")):
                rows = read(
                    os.path.join(
                        res, "cap", f"samples_stage1_{cond}_gsm8k.jsonl"
                    )
                )
                n = len({r["doc_id"] for r in rows})
                total = sum(r["exact_match"] for r in rows)
                scores[c] = (total - stage[cond]["gsm8k"] * n) / n
                ses[c] = math.sqrt(scores[c] * (1 - scores[c]) / n)
            metric = "exact_match,flexible-extract"
        out[t] = (metric, *scores, ses)
    return out


def table4(res):
    scores = capability_scores(res)
    if not scores:
        return
    print("Table 4. Capability cost of the cap (stage 1 sizes).\n")
    print("| Benchmark | Metric | Uncapped | Capped |")
    print("|---|---|---|---|")
    for t, (metric, a, b, _) in scores.items():
        print(f"| {t} | {metric} | {a:.3f} | {b:.3f} |")
    print()


def appendix_capping(res):
    sets = [
        ("A1. MHJ", os.path.join(res, "jb", "cap", "mhj"), "behavior"),
        ("A2. DAN", os.path.join(res, "dan3", "cap", "itw"), "behavior"),
    ]
    for title, stem, unit in sets:
        a = verdicts(stem + "_none.judged_v3.2.jsonl")
        b = verdicts(stem + "_capped.judged_v3.2.jsonl")
        meta = {r["id"]: r for r in read(stem + "_none.jsonl")}
        by_cat = collections.defaultdict(list)
        for i in a:
            if i in b:
                by_cat[meta[i]["category"]].append(i)
        print(f"Table {title}: harmful rate by category, uncapped vs capped.\n")
        print("| Category | n | Uncapped | Capped | Change, points (95% CI) |")
        print("|---|---|---|---|---|")
        rate = lambda ids, s: np.mean([s[i] in HARMFUL for i in ids])
        for cat, ids in sorted(by_cat.items(), key=lambda kv: -rate(kv[1], a)):
            pairs = [(a[i] in HARMFUL, b[i] in HARMFUL) for i in ids]
            diff = lambda p: 100 * np.mean([y - x for x, y in p])
            lo, hi = ci(pairs, diff, [meta[i][unit] for i in ids])
            print(
                f"| {cat} | {len(ids)} | {rate(ids, a):.0%} |"
                f" {rate(ids, b):.0%} | {diff(pairs):+.0f} ({lo:+.0f},"
                f" {hi:+.0f}) |"
            )
        print()


def appendix_harm_categories(res):
    sets = [
        ("Role openings", [os.path.join(res, "roles_q2", "harm_wild")]),
        (
            "Real openings",
            [
                os.path.join(res, "harmfloor", s)
                for s in ("harm_wild", "harm_wild_lowest", "harm_wild_k1")
            ],
        ),
    ]
    counts = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0])
    )
    for name, stems in sets:
        for stem in stems:
            score = verdicts(stem + ".judged_v3.2.jsonl")
            for r in read(stem + ".jsonl"):
                if r["id"] in score:
                    c = counts[r["category"]][name]
                    c[0] += 1
                    c[1] += score[r["id"]] in HARMFUL
    print("Table A3. Harmful rate by CategoricalHarmfulQA category.\n")
    print(
        "| Category | Role openings: n, harmful | Real openings: n, harmful |"
    )
    print("|---|---|---|")
    total = lambda v: sum(x[1] for x in v.values()) / sum(
        x[0] for x in v.values()
    )
    for cat, v in sorted(counts.items(), key=lambda kv: -total(kv[1])):
        cells = [f"{v[n][0]}, {v[n][1] / v[n][0]:.1%}" for n, _ in sets]
        print(f"| {cat} | {cells[0]} | {cells[1]} |")
    print()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results")
    for table in (
        table1,
        table1b,
        table1c,
        table2,
        table3,
        table4,
        appendix_capping,
        appendix_harm_categories,
    ):
        table(res)


if __name__ == "__main__":
    main()
