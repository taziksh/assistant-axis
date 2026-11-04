"""Redraw the paper's figures with our data.

Usage: figures.py [RESULTS_DIR] [OUT_DIR]   (defaults: wild/results, wild/figures)
Writes PNGs:
- fig25_drift.png: the paper's Fig. 25 (drift by turn, Llama 3.3 70B). Left:
  our real WildChat chats (same chats at every turn, mean and 95% CI) against
  the paper's four domains. Right: our projection of the paper's two released
  Llama conversations against the paper's curve for the same domain and
  simulated user.
- fig8_harm_vs_drift.png: the paper's Fig. 8 (harmful rate vs the first turn's
  projection, one dot per opening) with CategoricalHarmfulQA questions. The
  paper's figure is Qwen 3 32B on a raw projection scale, so only the shape
  compares, not the x values.
- fig10_capping.png: the paper's Fig. 10 (Llama row): harmful rate and
  capability scores, uncapped vs capped.
Paper curves come from paper_fig25.json (digitized from the figure); paper bars
are the values printed on Fig. 10.
"""

import collections
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tables import capability_scores
from tables import ci
from tables import HARMFUL
from tables import openings
from tables import paper_fig25
from tables import read
from tables import RELEASED_USER
from tables import verdicts

OURS = "#1f4e79"
PAPER = "#9a9a9a"
DOMAINS = {
    "coding": "#4c72b0",
    "writing": "#55a868",
    "therapy": "#c44e52",
    "philosophy": "#8172b2",
}
HARM = "#d9534f"
HARM_CAPPED = "#ee9a99"
BASE = "#5c5b57"
CAPPED = "#b1afa7"
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


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_drift(res, out):
    by_chat = collections.defaultdict(dict)
    for r in read(os.path.join(res, "drift_turns", "projections.jsonl")):
        if r["proj"] is not None:
            by_chat[r["hash"]][r["k"]] = r["proj"]
    turns = list(range(1, 13))
    chats = [c for c in by_chat.values() if all(k in c for k in turns)]
    paper = paper_fig25()
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 4.2), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    for domain, color in DOMAINS.items():
        curves = [v for (d, _), v in paper.items() if d == domain]
        mean = [
            np.mean([c[t] for c in curves if c[t] is not None])
            for t in range(15)
        ]
        left.plot(
            range(1, 16),
            mean,
            "-o",
            ms=3,
            lw=1.2,
            color=color,
            alpha=0.55,
            label=f"Paper: {domain}",
        )
    means = [np.mean([c[k] for c in chats]) for k in turns]
    bands = [ci([c[k] for c in chats], np.mean) for k in turns]
    left.fill_between(
        turns,
        [b[0] for b in bands],
        [b[1] for b in bands],
        color=OURS,
        alpha=0.15,
        lw=0,
    )
    left.plot(
        turns,
        means,
        "-o",
        ms=4,
        lw=2.2,
        color=OURS,
        label=f"Ours: real WildChat ({len(chats)} chats)",
    )
    left.set_title("Real chats vs the paper's simulated conversations")
    left.set_xlabel("Turn")
    left.set_ylabel("Assistant Axis projection")
    left.legend(frameon=False, fontsize=8, loc="lower left")

    scale = collections.defaultdict(dict)
    for r in read(os.path.join(res, "drift_scale", "projections.jsonl")):
        scale[r["hash"]][r["k"]] = r["proj"]
    for domain, user in RELEASED_USER.items():
        color = DOMAINS[domain]
        curve = paper[(domain, user)]
        right.plot(
            [t + 1 for t, v in enumerate(curve) if v is not None],
            [v for v in curve if v is not None],
            "-o",
            ms=3,
            lw=1.2,
            color=color,
            alpha=0.55,
            label=f"Paper: {domain} ({user}), mean of 100",
        )
        v = scale[f"paper-{domain}"]
        ks = sorted(v)
        right.plot(
            ks,
            [v[k] for k in ks],
            "--s",
            ms=3.5,
            lw=1.6,
            color=color,
            label=f"Ours: their released {domain} conversation",
        )
    right.set_title("Scale check: our code on the paper's conversations")
    right.set_xlabel("Turn")
    right.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in (left, right):
        style(ax)
        ax.set_xticks([1, 5, 10, 15])
        ax.set_ylim(-0.8, 1.6)
    fig.suptitle(
        "Persona drift, Llama 3.3 70B (the paper's Fig. 25 with our data)",
        fontsize=12,
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_harm(res, out):
    real = os.path.join(res, "harmfloor")
    sets = [
        (
            "Paper's role prompts + CatQA question",
            [os.path.join(res, "roles_q2", "harm_wild")],
        ),
        (
            "Real WildChat openings + CatQA question",
            [
                os.path.join(real, s)
                for s in ("harm_wild", "harm_wild_lowest", "harm_wild_k1")
            ],
        ),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.2), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    for ax, (name, stems) in zip(axes, sets):
        ops = [
            o
            for s in stems
            for o in openings(s + ".jsonl", s + ".judged_v3.2.jsonl")
        ]
        x = np.array([o[0] for o in ops])
        y = np.array([o[1] for o in ops])
        rho = lambda o: spearmanr([p[0] for p in o], [p[1] for p in o])[0]
        lo, hi = ci(ops, rho)
        ax.scatter(x, y, s=12, alpha=0.45, color=OURS, lw=0)
        ax.set_title(f"{name}\n{len(ops)} openings", fontsize=10)
        ax.text(
            0.98,
            0.95,
            f"Spearman ρ = {rho(ops):+.2f} ({lo:+.2f}, {hi:+.2f})\n"
            "paper: |r| = 0.39–0.52",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.set_xlabel(
            "Assistant Axis projection of the opening reply\n"
            "← role-playing     Assistant-like →"
        )
        style(ax)
    axes[0].set_ylabel("Rate of harmful responses")
    fig.suptitle(
        "Harmful rate vs drift, Llama 3.3 70B (the paper's Fig. 8 with"
        " our data)",
        fontsize=12,
        y=1.06,
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_capping(res, out):
    sets = [
        ("MHJ", os.path.join(res, "jb", "cap", "mhj")),
        ("DAN", os.path.join(res, "dan3", "cap", "itw")),
        (
            "Shah et al.\n(regenerated)",
            os.path.join(res, "shah", "cap", "shah"),
        ),
    ]
    jail = []
    for name, stem in sets:
        a = verdicts(stem + "_none.judged_v3.2.jsonl")
        b = verdicts(stem + "_capped.judged_v3.2.jsonl")
        ids = [i for i in a if i in b]
        jail.append((
            name,
            np.mean([a[i] in HARMFUL for i in ids]),
            np.mean([b[i] in HARMFUL for i in ids]),
        ))
    caps = capability_scores(res)
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 4.2),
        gridspec_kw={"width_ratios": [4, 4], "wspace": 0.15},
    )
    groups = [("Paper\n(persona jailbreaks)", *PAPER_FIG10["jailbreak"])]
    groups += jail
    w = 0.38
    for g, (name, a, b) in enumerate(groups):
        for off, v, c, lab in (
            (-w / 2, a, HARM, "No cap"),
            (w / 2, b, HARM_CAPPED, "Capped"),
        ):
            ax1.bar(g + off, v, w, color=c, label=lab if g == 0 else None)
            ax1.text(g + off, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax1.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=9)
    ax1.set_ylabel("Jailbreak rate (harmful responses)")
    ax1.set_ylim(0, 0.8)
    ax1.legend(
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
    )
    names = [t for t in BENCH_NAMES if t in caps]
    for g, t in enumerate(names):
        _, a, b = caps[t]
        pa, pb = PAPER_FIG10[t]
        div = 100 if t == "eq_bench" else 1  # EQ-Bench is on a 0-100 scale.
        for off, v, p, c, lab in (
            (-w / 2, a, pa, BASE, "No cap"),
            (w / 2, b, pb, CAPPED, "Capped"),
        ):
            ax2.bar(g + off, v / div, w, color=c, label=lab if g == 0 else None)
            ax2.text(
                g + off,
                v / div + 0.015,
                f"{v:.1f}" if div > 1 else f"{v:.2f}",
                ha="center",
                fontsize=8,
            )
            ax2.plot(
                g + off,
                p / div,
                marker="_",
                ms=16,
                mew=2,
                color="black",
                label="Paper" if g == 0 and off < 0 else None,
            )
    ax2.set_xticks(range(len(names)), [BENCH_NAMES[t] for t in names])
    ax2.set_ylabel("Capability score (EQ-Bench / 100)")
    ax2.set_ylim(0, 1.1)
    ax2.legend(
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
    )
    for ax in (ax1, ax2):
        style(ax)
    fig.suptitle(
        "Activation capping, Llama 3.3 70B (the paper's Fig. 10 with our data)",
        fontsize=12,
        y=1.04,
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "figures")
    os.makedirs(out, exist_ok=True)
    fig_drift(res, os.path.join(out, "fig25_drift.png"))
    fig_harm(res, os.path.join(out, "fig8_harm_vs_drift.png"))
    fig_capping(res, os.path.join(out, "fig10_capping.png"))
    print(f"wrote 3 figures to {out}")


if __name__ == "__main__":
    main()
