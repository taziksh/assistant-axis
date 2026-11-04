"""Redraw the paper's figures with our data.

Usage: figures.py [RESULTS_DIR] [OUT_DIR]   (defaults: wild/results, wild/figures)
Ours is always dark red and the paper always grey; error bars and bands on our
numbers are 95% CIs. Writes PNGs:
- fig25_drift.png: the paper's Fig. 25 (drift by turn, Llama 3.3 70B). (a) Our
  real WildChat chats (the same chats at every turn) against the paper's four
  domains. (b, c) Our projection of the paper's two released Llama
  conversations against the paper's curve for the same domain and simulated
  user.
- fig8_harm_vs_drift.png: the paper's Fig. 8 (harmful rate vs the opening's
  projection, one dot per opening). (a) The paper's own figure, which is Qwen 3
  32B on a raw projection scale, so it sits beside ours rather than on the same
  axes. (b, c) Ours with CategoricalHarmfulQA questions, with the mean harmful
  rate in projection bins.
- fig10_capping.png: the paper's Fig. 10 (Llama row): harmful rate and
  capability scores, uncapped vs capped.
Paper curves come from paper_fig25.json (digitized from the figure); paper bars
are the values printed on Fig. 10; the paper's Fig. 8 is downloaded from arXiv.
"""

import collections
import os
import sys
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
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

HERE = os.path.dirname(os.path.abspath(__file__))
OURS = "#7a1010"
OURS_LIGHT = "#d98c8c"
PAPER = "#6b6b6b"
PAPER_LIGHT = "#bdbdbd"
PAPER_FIG8_URL = (
    "https://arxiv.org/html/2601.10387v1/img/Axis_Assistant-Fig5.png"
)
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
DOMAIN_MARKERS = {
    "coding": "o",
    "writing": "s",
    "therapy": "^",
    "philosophy": "D",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def title(ax, letter, text, **kw):
    ax.set_title(f"({letter}) {text}", loc="left", **kw)


def fig_drift(res, out):
    by_chat = collections.defaultdict(dict)
    for r in read(os.path.join(res, "drift_turns", "projections.jsonl")):
        if r["proj"] is not None:
            by_chat[r["hash"]][r["k"]] = r["proj"]
    turns = list(range(1, 13))
    chats = [c for c in by_chat.values() if all(k in c for k in turns)]
    paper = paper_fig25()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4),
        sharey=True,
        gridspec_kw={"width_ratios": [1.5, 1, 1], "wspace": 0.12},
    )
    ax = axes[0]
    for domain, marker in DOMAIN_MARKERS.items():
        curves = [v for (d, _), v in paper.items() if d == domain]
        mean = [
            np.mean([c[t] for c in curves if c[t] is not None])
            for t in range(15)
        ]
        ax.plot(
            range(1, 16),
            mean,
            "-",
            marker=marker,
            ms=3.5,
            lw=1.1,
            color=PAPER_LIGHT,
            mfc=PAPER,
            mec=PAPER,
        )
        ax.text(15.4, mean[-1], domain, color=PAPER, va="center", fontsize=8.5)
    bands = [ci([c[k] for c in chats], np.mean) for k in turns]
    ax.fill_between(
        turns,
        [b[0] for b in bands],
        [b[1] for b in bands],
        color=OURS,
        alpha=0.15,
        lw=0,
    )
    ax.plot(
        turns,
        [np.mean([c[k] for c in chats]) for k in turns],
        "-o",
        ms=4,
        lw=2.2,
        color=OURS,
        label=f"ours: real WildChat chats (n = {len(chats)})",
    )
    ax.plot(
        [],
        [],
        "-o",
        ms=3.5,
        color=PAPER_LIGHT,
        mfc=PAPER,
        mec=PAPER,
        label="paper: simulated conversations, by domain",
    )
    title(ax, "a", "Real chats vs the paper's domains")
    ax.set_ylabel("Assistant Axis projection")
    ax.legend(frameon=False, loc="lower left")
    ax.set_xlim(0.5, 17.5)
    scale = collections.defaultdict(dict)
    for r in read(os.path.join(res, "drift_scale", "projections.jsonl")):
        scale[r["hash"]][r["k"]] = r["proj"]
    for ax, letter, (domain, user) in zip(
        axes[1:], "bc", RELEASED_USER.items()
    ):
        curve = paper[(domain, user)]
        ax.plot(
            [t + 1 for t, v in enumerate(curve) if v is not None],
            [v for v in curve if v is not None],
            "-o",
            ms=3.5,
            lw=1.4,
            color=PAPER,
            label=f"paper: mean of 100 ({user} as user)",
        )
        v = scale[f"paper-{domain}"]
        ks = sorted(v)
        ax.plot(
            ks,
            [v[k] for k in ks],
            "-o",
            ms=4,
            lw=1.8,
            color=OURS,
            label="ours: their one released conversation",
        )
        title(ax, letter, f"Scale check: {domain}")
        ax.legend(frameon=False, loc="lower left")
    for ax in axes:
        style(ax)
        ax.set_xlabel("Turn")
        ax.set_xticks([1, 5, 10, 15])
        ax.set_ylim(-0.8, 1.6)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def paper_fig8():
    path = os.path.join(HERE, "data", "paper", "fig8.png")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(PAPER_FIG8_URL, path)
    return mpimg.imread(path)


def binned(x, y, bins=8):
    """Mean of y and its 95% CI in equal-count bins of x."""
    order = np.argsort(x)
    out = []
    for idx in np.array_split(order, bins):
        lo, hi = ci(list(y[idx]), np.mean)
        out.append((x[idx].mean(), y[idx].mean(), lo, hi))
    return np.array(out)


def fig_harm(res, out):
    real = os.path.join(res, "harmfloor")
    sets = [
        (
            "ours: paper's role-prompt openings",
            [os.path.join(res, "roles_q2", "harm_wild")],
        ),
        (
            "ours: real WildChat openings",
            [
                os.path.join(real, s)
                for s in ("harm_wild", "harm_wild_lowest", "harm_wild_k1")
            ],
        ),
    ]
    fig = plt.figure(figsize=(11, 9.5))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.05, 1], hspace=0.28, wspace=0.08
    )
    ax = fig.add_subplot(grid[0, :])
    ax.imshow(paper_fig8())
    ax.axis("off")
    title(
        ax,
        "a",
        "Paper: Qwen 3 32B, 440 questions (raw projection scale)",
        color=PAPER,
    )
    first = None
    for g, letter, (name, stems) in zip((0, 1), "bc", sets):
        ax = fig.add_subplot(grid[1, g], sharey=first)
        first = first or ax
        ops = [
            o
            for s in stems
            for o in openings(s + ".jsonl", s + ".judged_v3.2.jsonl")
        ]
        x = np.array([o[0] for o in ops])
        y = np.array([o[1] for o in ops])
        jitter = np.random.default_rng(0).uniform(-0.006, 0.006, len(y))
        ax.scatter(x, y + jitter, s=9, alpha=0.3, color=OURS_LIGHT, lw=0)
        b = binned(x, y)
        ax.errorbar(
            b[:, 0],
            b[:, 1],
            yerr=[b[:, 1] - b[:, 2], b[:, 3] - b[:, 1]],
            fmt="-o",
            ms=4,
            lw=1.8,
            capsize=2,
            color=OURS,
            label="mean in 8 bins (95% CI)",
        )
        rho = lambda o: spearmanr([p[0] for p in o], [p[1] for p in o])[0]
        lo, hi = ci(ops, rho)
        ax.text(
            0.98,
            0.97,
            f"ours: ρ = {rho(ops):+.2f} ({lo:+.2f}, {hi:+.2f})\n"
            "paper: |r| = 0.39–0.52",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
        )
        title(ax, letter, f"{name[0].upper()}{name[1:]} ({len(ops)})")
        ax.set_xlabel("Projection of the opening reply")
        if g == 0:
            ax.set_ylabel("Rate of harmful responses")
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.86))
        else:
            ax.tick_params(labelleft=False)
        style(ax)
    first.set_ylim(-0.02, 0.32)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def bar_pair(ax, x, w, a, b, colors, label=None, errs=None):
    """Two bars (uncapped, capped) centered at x, with values printed above."""
    for off, v, c, e, cond in (
        (-w / 2, a, colors[0], errs[0] if errs else None, "no cap"),
        (w / 2, b, colors[1], errs[1] if errs else None, "capped"),
    ):
        ax.bar(
            x + off,
            v,
            w,
            color=c,
            yerr=None if e is None else [[v - e[0]], [e[1] - v]],
            capsize=2,
            error_kw={"lw": 1},
            label=f"{label}, {cond}" if label else None,
        )
        top = v if e is None else e[1]
        ax.text(x + off, top + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)


def fig_capping(res, out):
    sets = [
        ("MHJ", os.path.join(res, "jb", "cap", "mhj"), "behavior"),
        ("DAN", os.path.join(res, "dan3", "cap", "itw"), "behavior"),
        (
            "Shah et al.\n(regenerated)",
            os.path.join(res, "shah", "cap", "shah"),
            "category",
        ),
    ]
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 4.2),
        gridspec_kw={"wspace": 0.15, "width_ratios": [1, 1.25]},
    )
    w = 0.38
    bar_pair(
        ax1,
        0,
        w,
        *PAPER_FIG10["jailbreak"],
        (PAPER, PAPER_LIGHT),
        label="paper",
    )
    for g, (name, stem, unit) in enumerate(sets, 1):
        a = verdicts(stem + "_none.judged_v3.2.jsonl")
        b = verdicts(stem + "_capped.judged_v3.2.jsonl")
        meta = {r["id"]: r for r in read(stem + "_none.jsonl")}
        ids = [i for i in a if i in b]
        units = [meta[i][unit] for i in ids]
        rates, errs = [], []
        for v in (a, b):
            hits = [float(v[i] in HARMFUL) for i in ids]
            rates.append(np.mean(hits))
            errs.append(ci(hits, np.mean, units))
        bar_pair(
            ax1,
            g,
            w,
            *rates,
            (OURS, OURS_LIGHT),
            label="ours" if g == 1 else None,
            errs=errs,
        )
    ax1.set_xticks(
        range(len(sets) + 1),
        ["Paper's persona\njailbreaks"] + [s[0] for s in sets],
    )
    ax1.set_ylabel("Jailbreak rate (harmful responses)")
    ax1.set_ylim(0, 0.8)
    title(ax1, "a", "Jailbreaks")
    ax1.legend(frameon=False, ncol=2, loc="upper right")
    caps = capability_scores(res)
    names = [t for t in BENCH_NAMES if t in caps]
    w = 0.2
    for g, t in enumerate(names):
        _, a, b, ses = caps[t]
        div = 100 if t == "eq_bench" else 1  # EQ-Bench is on a 0-100 scale.
        pa, pb = PAPER_FIG10[t]
        for off, v, c, se, lab in (
            (-1.5 * w, pa, PAPER, None, "paper, no cap"),
            (-0.5 * w, pb, PAPER_LIGHT, None, "paper, capped"),
            (0.5 * w, a, OURS, ses[0], "ours, no cap"),
            (1.5 * w, b, OURS_LIGHT, ses[1], "ours, capped"),
        ):
            ax2.bar(
                g + off,
                v / div,
                w,
                color=c,
                yerr=None if se is None else 1.96 * se / div,
                capsize=2,
                error_kw={"lw": 1},
                label=lab if g == 0 else None,
            )
            ax2.text(
                g + off,
                (v + (1.96 * se if se else 0)) / div + 0.012,
                f"{v:.1f}" if div > 1 else f"{v:.2f}",
                ha="center",
                fontsize=6.5,
            )
    ax2.set_xticks(range(len(names)), [BENCH_NAMES[t] for t in names])
    ax2.set_ylabel("Score (EQ-Bench ÷ 100)")
    ax2.set_ylim(0, 1.15)
    title(ax2, "b", "Capabilities")
    ax2.legend(frameon=False, ncol=4, loc="upper center", fontsize=7.5)
    for ax in (ax1, ax2):
        style(ax)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    fig_drift(res, os.path.join(out, "fig25_drift.png"))
    fig_harm(res, os.path.join(out, "fig8_harm_vs_drift.png"))
    fig_capping(res, os.path.join(out, "fig10_capping.png"))
    print(f"wrote 3 figures to {out}")


if __name__ == "__main__":
    main()
