"""Drift by domain with the context-length controls (results of go_domain.sh).

Usage: domain_drift.py RUN_DIR [RUN_DIR ...]
RUN_DIR holds {real,filler,nohist}/{selected,projections}.jsonl;
real/selected.jsonl carries each chat's classifier label in "domain".

Per domain and user turn k:
- real: projection of the reply after the chat's own history.
- filler: reply to the same message after k-1 unrelated exchanges.
- nohist: reply to the same message with no history.
All are reported as the change from the same chat's turn-1 projection (turn 1
has no history, so it is the same in all three conditions). A chat contributes
to a cell only if it has both values.

"Specific to the chat" is real minus filler at the same turn of the same chat:
the part of the drift that unrelated history of the same length does not
produce. It is averaged over each chat's turns 2-12 first, then over chats.

CIs are 95% percentile bootstraps over chats (2,000 draws, seed 0).
"""

import collections
import json
import os
import sys

import numpy as np

DOMAINS = [("coding", "Coding"), ("writing", "Writing"), ("therapy", "Therapy"),
           ("ai", "AI philosophy")]
TURNS = [2, 4, 6, 9, 12]
DRAWS = 2000


def rj(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load(run):
    dom = {r["hash"]: r["domain"] for r in rj(os.path.join(run, "real", "selected.jsonl"))}
    proj = {"real": {}, "filler": {}, "nohist": {}}
    for r in rj(os.path.join(run, "real", "projections.jsonl")):
        if r["proj"] is not None:
            proj["real"][(r["hash"], r["k"])] = r["proj"]
    for cond, tag in (("filler", "-f"), ("nohist", "-n")):
        for r in rj(os.path.join(run, cond, "projections.jsonl")):
            if r["proj"] is None:
                continue
            target, k = r["hash"].rsplit(tag, 1)
            proj[cond][(target, int(k))] = r["proj"]
    return dom, proj


def ci(values):
    """Mean and 95% bootstrap CI over chats; None if fewer than 2."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return None
    rng = np.random.default_rng(0)
    means = v[rng.integers(0, len(v), (DRAWS, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return v.mean(), lo, hi


def cell(values, d):
    c = ci(values)
    if c is None:
        return "–"
    return f"{c[0]:+.{d}f} ({c[1]:+.{d}f}, {c[2]:+.{d}f})"


def report(run):
    dom, proj = load(run)
    d = 1 if "qwen" in run.lower() else 2  # Qwen's scale is about 10x Llama's.
    chats = collections.defaultdict(list)
    for h, name in dom.items():
        if (h, 1) in proj["real"]:
            chats[name].append(h)
    print(f"\n## {run}\n")
    print("| Domain | Chats | Turn 1 projection | Chats reaching turn 12 |")
    print("|---|---|---|---|")
    for key, label in DOMAINS:
        hs = chats[key]
        t1 = ci([proj["real"][(h, 1)] for h in hs])
        n12 = sum((h, 12) in proj["real"] for h in hs)
        print(f"| {label} | {len(hs)} | {t1[0]:.{d}f} ({t1[1]:.{d}f}, {t1[2]:.{d}f}) | {n12} |")

    for cond, title in (("real", "Real history"), ("filler", "Unrelated history"),
                        ("nohist", "No history")):
        print(f"\n{title}: change from turn 1 (95% CI), n chats\n")
        print("| Turn | " + " | ".join(label for _, label in DOMAINS) + " |")
        print("|---|" + "---|" * len(DOMAINS))
        for k in TURNS:
            row = []
            for key, _ in DOMAINS:
                vals = [proj[cond][(h, k)] - proj["real"][(h, 1)]
                        for h in chats[key] if (h, k) in proj[cond]]
                row.append(f"{cell(vals, d)}, n={len(vals)}")
            print(f"| {k} | " + " | ".join(row) + " |")

    print("\nAveraged over each chat's turns 2-12 (95% CI)\n")
    print("| Domain | Chats | Real history | Unrelated history | No history |"
          " Specific to the chat (real − unrelated) |")
    print("|---|---|---|---|---|---|")
    spec = {}
    for key, label in DOMAINS:
        per = {"real": [], "filler": [], "nohist": [], "spec": []}
        for h in chats[key]:
            base = proj["real"][(h, 1)]
            ks = [k for k in range(2, 13)
                  if all((h, k) in proj[c] for c in ("real", "filler", "nohist"))]
            if not ks:
                continue
            for c in ("real", "filler", "nohist"):
                per[c].append(np.mean([proj[c][(h, k)] - base for k in ks]))
            per["spec"].append(np.mean([proj["real"][(h, k)] - proj["filler"][(h, k)] for k in ks]))
        spec[key] = per["spec"]
        print(f"| {label} | {len(per['real'])} | " + " | ".join(
            cell(per[c], d) for c in ("real", "filler", "nohist", "spec")) + " |")

    print("\nDifference between domains in the chat-specific part (95% CI)\n")
    print("| Comparison | Difference |")
    print("|---|---|")
    rng = np.random.default_rng(0)
    pairs = [("therapy", "coding"), ("therapy", "writing"), ("ai", "coding"),
             ("ai", "writing"), ("therapy", "ai"), ("coding", "writing")]
    names = dict(DOMAINS)
    for a, b in pairs:
        x, y = np.asarray(spec[a]), np.asarray(spec[b])
        draws = (x[rng.integers(0, len(x), (DRAWS, len(x)))].mean(axis=1)
                 - y[rng.integers(0, len(y), (DRAWS, len(y)))].mean(axis=1))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(f"| {names[a]} − {names[b]} | {x.mean() - y.mean():+.{d}f} ({lo:+.{d}f}, {hi:+.{d}f}) |")


def main():
    for run in sys.argv[1:]:
        report(run)


if __name__ == "__main__":
    main()
