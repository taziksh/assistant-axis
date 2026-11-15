"""Tables from capfire.py: how often the cap would bind on chat replies, by turn.

Usage: capfire_score.py DIR [--jb-only]
DIR holds {real,filler,nohist}/capfire.jsonl and jb/capfire.jsonl for one model
(go_capfire.sh's layout). Per reply, three measures:
- share: reply tokens past the cap, as a share, averaged over the capped layers
  (capdiag.py's measure);
- excess: the amount past the cap per reply token, averaged over the capped
  layers (what the cap would remove; in the model's activation units, so not
  comparable across models);
- any: share of reply tokens past the cap at one or more layers
  (activations.py's fire_frac).
Prints:
- the jailbreak reference, pooled over tokens as capdiag.py does and as a mean
  over attacks;
- by turn and history condition, the mean over chats with a 95% bootstrap CI
  (5,000 draws over chats, seed 0);
- turn 12 minus the same chat's turn 1 (real history, cap off), per condition.
Where DIR/<condition>/projections.jsonl exists, checks `any` against its
fire_frac for the same replies (they come from the same forward pass).
Turn: k for real history; the -f<k> / -n<k> suffix of the hash for unrelated
history and no history (lenctrl_build.py). Turn 1 has no history in any
condition, so only `real` has it.
"""

import collections
import json
import os
import sys

import numpy as np

DRAWS = 5000
CONDS = ["real", "filler", "nohist"]
MEASURES = ["share", "excess", "any"]


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def measures(r):
    n = r["n_tok"]
    return {
        "share": float(np.mean(r["over"])) / n,
        "excess": float(np.mean(r["excess"])) / n,
        "any": r["any_over"] / n,
    }


def boot(values):
    v = np.asarray(values, float)
    idx = np.random.default_rng(0).integers(0, len(v), (DRAWS, len(v)))
    return v.mean(), *np.percentile(v[idx].mean(1), [2.5, 97.5])


def fmt(values, digits):
    if len(values) < 2:
        return "–"
    m, lo, hi = boot(values)
    return f"{m:.{digits}f} ({lo:.{digits}f}, {hi:.{digits}f})"


def jb(d):
    path = os.path.join(d, "jb", "capfire.jsonl")
    if not os.path.exists(path):
        print("No jailbreak reference (jb/capfire.jsonl).\n")
        return
    rows = [r for r in read(path) if r["n_tok"]]
    n_tok = sum(r["n_tok"] for r in rows)
    over = np.sum([r["over"] for r in rows], 0) / n_tok  # Per layer, pooled.
    excess = np.sum([r["excess"] for r in rows], 0) / n_tok
    print(f"Jailbreak reference: {len(rows)} attacks, layers {rows[0]['layers']}.")
    print(
        f"- pooled over tokens (capdiag.py's way): share {over.mean():.3f}"
        f" (per layer {over.min():.3f} to {over.max():.3f}), excess"
        f" {excess.mean():.2f}, any {sum(r['any_over'] for r in rows) / n_tok:.3f}"
    )
    ms = [measures(r) for r in rows]
    print(
        "- mean over attacks (95% CI): "
        + ", ".join(
            f"{m} {fmt([x[m] for x in ms], 2 if m == 'excess' else 3)}"
            for m in MEASURES
        )
        + "\n"
    )


def load(d, cond):
    """chat -> turn -> measures, plus the fire_frac check."""
    path = os.path.join(d, cond, "capfire.jsonl")
    if not os.path.exists(path):
        return None
    out = collections.defaultdict(dict)
    rows = read(path)
    for r in rows:
        if not r["n_tok"]:
            continue
        if cond == "real":
            chat, k = r["hash"], r["k"]
        else:
            chat, suffix = r["hash"].rsplit("-", 1)
            k = int(suffix[1:])
        out[chat][k] = measures(r)
    proj = os.path.join(d, cond, "projections.jsonl")
    if os.path.exists(proj):
        fire = {(p["hash"], p["k"]): p["fire_frac"] for p in read(proj)}
        diffs = [
            abs(r["any_over"] / r["n_tok"] - fire[(r["hash"], r["k"])])
            for r in rows
            if r["n_tok"] and fire.get((r["hash"], r["k"])) is not None
        ]
        if diffs:
            print(
                f"check {cond}: `any` vs saved fire_frac on {len(diffs)} replies,"
                f" largest difference {max(diffs):.4f}"
            )
    return out


def main():
    d = sys.argv[1]
    jb(d)
    if "--jb-only" in sys.argv:
        return
    data = {c: load(d, c) for c in CONDS}
    data = {c: v for c, v in data.items() if v}
    print()
    for m in MEASURES:
        digits = 2 if m == "excess" else 3
        print(f"`{m}` by turn, mean over chats (95% CI):\n")
        print("| Turn | " + " | ".join(f"{c} | n" for c in data) + " |")
        print("|---|" + "---|---|" * len(data))
        for k in range(1, 13):
            cells = []
            for c in data:
                vals = [t[k][m] for t in data[c].values() if k in t]
                cells += [fmt(vals, digits), str(len(vals))]
            print(f"| {k} | " + " | ".join(cells) + " |")
        print()
    if "real" in data:
        print("Turn 12 minus the same chat's turn 1 (real history), mean (95% CI):\n")
        print("| Measure | " + " | ".join(f"{c} | n" for c in data) + " |")
        print("|---|" + "---|---|" * len(data))
        for m in MEASURES:
            cells = []
            for c in data:
                vals = [
                    t[12][m] - data["real"][chat][1][m]
                    for chat, t in data[c].items()
                    if 12 in t and 1 in data["real"].get(chat, {})
                ]
                cells += [fmt(vals, 2 if m == "excess" else 3), str(len(vals))]
            print(f"| {m} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
