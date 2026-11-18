"""Statistics for the Section 4.3 rerun (q43b_build.py): opening projection
against the rate of harmful second-turn replies.

Usage: q43b_stats.py [RUN_DIR] [PAPER_PROJ_JSONL]
Reads RUN_DIR/harm_block1.jsonl, its .judged_v3.2_lu.jsonl verdicts and
harm_openings.jsonl (default results/q43b), and the paper's own projections of
the same openings (default results/q43/paper_proj.jsonl). Harmful is the
authors' definition (tables.HARMFUL). Intervals are 95% bootstraps over roles,
since a role's openings share a system prompt.
"""

import collections
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

from tables import HARMFUL, read, verdicts

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results/q43b")
    paper = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(HERE, "results/q43/paper_proj.jsonl")
    )
    score = verdicts(os.path.join(run, "harm_block1.judged_v3.2_lu.jsonl"))
    openings = {r["hash"]: r for r in read(os.path.join(run, "harm_openings.jsonl"))}
    theirs = {r["hash"]: r for r in read(paper)}
    by_opening = collections.defaultdict(dict)
    for r in read(os.path.join(run, "harm_block1.jsonl")):
        by_opening[r["hash"]][r["question_id"]] = score[r["id"]] in HARMFUL
    hashes = sorted(by_opening)
    questions = sorted(by_opening[hashes[0]])
    m = np.array([[by_opening[h][q] for q in questions] for h in hashes], float)
    rate = m.mean(1)
    proj = np.array([openings[h]["proj"] for h in hashes])
    role = np.array([openings[h]["role"] for h in hashes])
    groups = [np.where(role == r)[0] for r in sorted(set(role))]
    rng = np.random.default_rng(0)
    print(
        f"{len(hashes)} openings, {len(groups)} roles, {len(questions)} questions;"
        f" {100 * m.mean():.1f}% of replies harmful"
    )

    def interval(x, stat):
        draws = []
        for _ in range(2000):
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[i] for i in pick])
            draws.append(stat(x[idx], rate[idx])[0])
        return np.percentile(draws, [2.5, 97.5])

    for name, x in (
        ("our axis projection", proj),
        ("paper's role PC1", np.array([theirs[h]["pc1"] for h in hashes])),
        ("paper's contrast vector", np.array([theirs[h]["contrast"] for h in hashes])),
    ):
        for label, stat in (("Pearson", pearsonr), ("Spearman", spearmanr)):
            lo, hi = interval(x, stat)
            print(f"{name}: {label} {stat(x, rate)[0]:+.2f} ({lo:+.2f}, {hi:+.2f})")

    # How stable the per-opening rate is: two random halves of the questions.
    half = len(questions) // 2
    halves = []
    for _ in range(1000):
        p = rng.permutation(len(questions))
        halves.append(
            pearsonr(m[:, p[:half]].mean(1), m[:, p[half:]].mean(1))[0]
        )
    r_half = np.mean(halves)
    print(
        f"split-half r = {r_half:.2f};"
        f" reliability of the full rate = {2 * r_half / (1 + r_half):.2f}"
    )

    # How much of the harm rate is the role itself.
    within = sum(((rate[g] - rate[g].mean()) ** 2).sum() for g in groups)
    total = ((rate - rate.mean()) ** 2).sum()
    print(f"share of harm-rate variance between roles: {1 - within / total:.2f}")
    print(f"share explained by the projection: {pearsonr(proj, rate)[0] ** 2:.2f}")
    edges = np.percentile(proj, [0, 25, 50, 75, 100])
    for k in range(4):
        q = (proj >= edges[k]) & (proj <= edges[k + 1])
        print(
            f"projection {edges[k]:+.0f} to {edges[k + 1]:+.0f}:"
            f" mean harmful rate {rate[q].mean():.3f}, highest {rate[q].max():.2f}"
        )


if __name__ == "__main__":
    main()
