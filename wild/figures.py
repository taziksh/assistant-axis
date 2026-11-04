"""Redraw the paper's figures with our data, one simple plot per figure.

Usage: figures.py [RESULTS_DIR] [OUT_DIR]   (defaults: wild/results, wild/figures)
Writes each figure as PNG (300 dpi) and SVG, with no titles (captions go in
the write-up):
- drift_real_chats: the paper's Fig. 25. Assistant Axis projection by turn for
  our real WildChat chats (the same 76 chats at every turn, 95% CI band) and
  the paper's coding conversations (its closest domain; mean over its three
  simulated users, digitized into paper_fig25.json).
- harm_vs_drift: the paper's Fig. 8. Mean harmful rate (CategoricalHarmfulQA)
  in 8 equal-count bins of the opening's projection, with 95% CIs, for role
  openings and real openings.
- capping_jailbreaks: the paper's Fig. 10, jailbreak half. Change in harmful
  rate from capping (capped minus uncapped, points), with 95% CIs clustered as
  in Table 3; the paper's change from its printed bars.
- capping_capability: the paper's Fig. 10, capability half. Change in each
  benchmark score from capping, ours (95% CI from the two runs' standard
  errors) next to the paper's.
Colors: the dataviz reference palette's first two slots, validated together.
"""

import collections
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tables import capability_scores
from tables import ci
from tables import HARMFUL
from tables import openings
from tables import paper_fig25
from tables import read
from tables import verdicts

HERE = os.path.dirname(os.path.abspath(__file__))
OURS = "#2a78d6"  # Blue (reference palette slot 1).
PAPER = "#eb6834"  # Orange (slot 2).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Llama 3.3 70B row of the paper's Fig. 10, as printed on the bars.
PAPER_FIG10 = {
    "jailbreak": (0.65, 0.33),
    "ifeval": (0.93, 0.93),
    "mmlu_pro": (0.71, 0.70),
    "gsm8k": (0.89, 0.89),
    "eq_bench": (83.1, 84.1),
}
BENCH_NAMES = {
    "ifeval": "IFEval",
    "mmlu_pro": "MMLU Pro",
    "gsm8k": "GSM8k",
    "eq_bench": "EQ-Bench",
}

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 1,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "text.color": INK,
    "svg.fonttype": "none",
})


def new_axes(size=(7, 4.2), grid="y"):
    fig, ax = plt.subplots(figsize=size)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    return fig, ax


def save(fig, out, name):
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(
            os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", **kw
        )
    plt.close(fig)


def end_label(ax, x, y, text):
    ax.annotate(
        text,
        (x, y),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        color=INK_2,
    )


def fig_drift(res, out):
    by_chat = collections.defaultdict(dict)
    for r in read(os.path.join(res, "drift_turns", "projections.jsonl")):
        if r["proj"] is not None:
            by_chat[r["hash"]][r["k"]] = r["proj"]
    turns = list(range(1, 13))
    chats = [c for c in by_chat.values() if all(k in c for k in turns)]
    coding = [v for (d, _), v in paper_fig25().items() if d == "coding"]
    paper = [np.mean([c[t - 1] for c in coding]) for t in turns]
    ours = [np.mean([c[k] for c in chats]) for k in turns]
    bands = [ci([c[k] for c in chats], np.mean) for k in turns]
    fig, ax = new_axes()
    ax.fill_between(
        turns,
        [b[0] for b in bands],
        [b[1] for b in bands],
        color=OURS,
        alpha=0.12,
        lw=0,
    )
    ax.plot(turns, paper, color=PAPER, lw=2)
    ax.plot(turns, ours, color=OURS, lw=2)
    end_label(ax, 12, paper[-1] + 0.03, "Paper: coding")
    end_label(ax, 12, ours[-1] - 0.03, "Ours: real chats")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Assistant Axis projection")
    ax.set_xticks([1, 3, 6, 9, 12])
    ax.set_xlim(0.6, 14.8)
    ax.set_ylim(0, 1.2)
    ax.spines["left"].set_bounds(0, 1.2)
    save(fig, out, "drift_real_chats")


def binned(x, y, bins=8):
    """Mean of y and its 95% CI in equal-count bins of x."""
    order = np.argsort(x)
    rows = []
    for idx in np.array_split(order, bins):
        lo, hi = ci(list(y[idx]), np.mean)
        rows.append((x[idx].mean(), y[idx].mean(), lo, hi))
    return np.array(rows)


def fig_harm(res, out):
    sets = [
        ("Role openings", OURS, ["roles_q2/harm_wild"]),
        (
            "Real chats",
            PAPER,
            [
                "harmfloor/harm_wild",
                "harmfloor/harm_wild_lowest",
                "harmfloor/harm_wild_k1",
            ],
        ),
    ]
    fig, ax = new_axes()
    for name, color, stems in sets:
        ops = [
            o
            for s in stems
            for o in openings(
                os.path.join(res, s + ".jsonl"),
                os.path.join(res, s + ".judged_v3.2.jsonl"),
            )
        ]
        b = binned(
            np.array([o[0] for o in ops]), 100 * np.array([o[1] for o in ops])
        )
        ax.fill_between(
            b[:, 0], b[:, 2], b[:, 3], color=color, alpha=0.12, lw=0
        )
        ax.plot(b[:, 0], b[:, 1], color=color, lw=2)
        end_label(ax, b[-1, 0], b[-1, 1], name)
    ax.set_xlabel(
        "Assistant Axis projection of the opening\n(left: more drifted)"
    )
    ax.set_ylabel("Harmful replies (%)")
    ax.set_ylim(0, 15)
    ax.set_xlim(None, ax.get_xlim()[1] + 0.9)
    save(fig, out, "harm_vs_drift")


def change_dots(ax, rows):
    """Horizontal dot-and-whisker rows: (label, value, (lo, hi) or None, color)."""
    for y, (label, v, err, color) in enumerate(rows):
        if err is not None:
            ax.hlines(y, err[0], err[1], color=color, lw=2.5, alpha=0.35)
        ax.plot(v, y, "o", ms=9, color=color, mec=SURFACE, mew=2, zorder=3)
        ax.annotate(
            f"{v:+.1f}",
            (v, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=INK_2,
            fontsize=11,
        )
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)


def fig_jailbreaks(res, out):
    a, b = PAPER_FIG10["jailbreak"]
    rows = [("Paper: persona\njailbreaks", 100 * (b - a), None, PAPER)]
    for name, stem, unit in (
        ("MHJ", "jb/cap/mhj", "behavior"),
        ("DAN", "dan3/cap/itw", "behavior"),
        ("Shah et al.\n(regenerated)", "shah/cap/shah", "category"),
    ):
        stem = os.path.join(res, stem)
        none = verdicts(stem + "_none.judged_v3.2.jsonl")
        capped = verdicts(stem + "_capped.judged_v3.2.jsonl")
        meta = {r["id"]: r for r in read(stem + "_none.jsonl")}
        ids = [i for i in none if i in capped]
        pairs = [(none[i] in HARMFUL, capped[i] in HARMFUL) for i in ids]
        diff = lambda p: 100 * np.mean([y - x for x, y in p])
        err = ci(pairs, diff, [meta[i][unit] for i in ids])
        rows.append((f"Ours: {name}", diff(pairs), err, OURS))
    fig, ax = new_axes(size=(7, 3.6), grid="x")
    change_dots(ax, rows)
    ax.set_xlabel("Change in harmful replies from capping (points)")
    ax.set_xlim(-40, 15)
    save(fig, out, "capping_jailbreaks")


def fig_capability(res, out):
    caps = capability_scores(res)
    names = [t for t in BENCH_NAMES if t in caps]
    if not names:
        return
    rows = []
    for t in names:
        _, a, b, ses = caps[t]
        scale = 1 if t == "eq_bench" else 100  # Points on a 0-100 scale.
        pa, pb = PAPER_FIG10[t]
        half = 1.96 * math.hypot(*ses) * scale
        d = (b - a) * scale
        rows.append(
            (f"{BENCH_NAMES[t]}: paper", (pb - pa) * scale, None, PAPER)
        )
        rows.append((f"{BENCH_NAMES[t]}: ours", d, (d - half, d + half), OURS))
    fig, ax = new_axes(size=(7, 4.6), grid="x")
    change_dots(ax, rows)
    for y in range(1, len(rows) - 1, 2):
        ax.axhline(y + 0.5, color=GRID, lw=1)
    ax.set_xlabel("Change in score from capping (points out of 100)")
    ax.set_xlim(-12, 12)
    save(fig, out, "capping_capability")


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    for old in os.listdir(out):
        os.remove(os.path.join(out, old))
    fig_drift(res, out)
    fig_harm(res, out)
    fig_jailbreaks(res, out)
    fig_capability(res, out)
    print(f"wrote 4 figures to {out}")


if __name__ == "__main__":
    main()
