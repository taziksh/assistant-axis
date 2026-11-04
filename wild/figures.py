"""Redraw the paper's figures with our data, one plot per figure.

Usage: figures.py [RESULTS_DIR] [OUT_DIR]   (defaults: wild/results, wild/figures)
Writes each figure as PNG (300 dpi) and SVG:
- drift_real_chats: the paper's Fig. 25 (drift by turn, Llama 3.3 70B). Our
  real WildChat chats (the same chats at every turn, mean and 95% CI) against
  the paper's four domains (digitized, paper_fig25.json). The scale check on
  the paper's own conversations is Table 1c.
- harm_vs_drift: the paper's Fig. 8 (harmful rate vs the opening's
  projection). Ours with CategoricalHarmfulQA questions: one dot per opening,
  and the mean in 8 equal-count bins with 95% CIs. The paper's figure is Qwen 3
  32B on a raw projection scale, so it is not redrawn on these axes.
- capping_jailbreaks, capping_capability: the paper's Fig. 10 (Llama row),
  uncapped vs capped, with the paper's printed values.
Colors: the first two slots of the dataviz reference palette (validated
together); the paper's domain curves are grey context.
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
from tables import verdicts

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES_1 = "#2a78d6"  # Blue.
SERIES_2 = "#eb6834"  # Orange.
CONTEXT = "#b9b7ae"  # Grey context lines.
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
    "eq_bench": "EQ-Bench (÷100)",
}

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 22,
    "axes.labelsize": 11,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 1,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "text.color": INK,
    "svg.fonttype": "none",
})


def new_axes(title, subtitle, size=(7.5, 4.6)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title)
    ax.text(
        0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        color=INK_2,
        fontsize=10,
        va="bottom",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    return fig, ax


def save(fig, out, name):
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(
            os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", **kw
        )
    plt.close(fig)


def fig_drift(res, out):
    by_chat = collections.defaultdict(dict)
    for r in read(os.path.join(res, "drift_turns", "projections.jsonl")):
        if r["proj"] is not None:
            by_chat[r["hash"]][r["k"]] = r["proj"]
    turns = list(range(1, 13))
    chats = [c for c in by_chat.values() if all(k in c for k in turns)]
    fig, ax = new_axes(
        "Real chats drift like the paper's coding conversations",
        "Llama 3.3 70B, Assistant Axis projection by turn (higher = more"
        " Assistant-like)",
    )
    paper = paper_fig25()
    for domain in ("coding", "writing", "therapy", "philosophy"):
        curves = [v for (d, _), v in paper.items() if d == domain]
        mean = [
            np.mean([c[t] for c in curves if c[t] is not None])
            for t in range(15)
        ]
        ax.plot(range(1, 16), mean, color=CONTEXT, lw=1.5, zorder=2)
        ax.text(15.3, mean[-1], domain, color=MUTED, va="center", fontsize=10)
    bands = [ci([c[k] for c in chats], np.mean) for k in turns]
    ax.fill_between(
        turns,
        [b[0] for b in bands],
        [b[1] for b in bands],
        color=SERIES_1,
        alpha=0.12,
        lw=0,
        zorder=3,
    )
    ax.plot(
        turns,
        [np.mean([c[k] for c in chats]) for k in turns],
        "-o",
        color=SERIES_1,
        lw=2,
        ms=6,
        mec=SURFACE,
        mew=1.5,
        zorder=4,
        label=f"Ours: real WildChat chats (n = {len(chats)}, 95% CI)",
    )
    ax.plot(
        [],
        [],
        color=CONTEXT,
        lw=1.5,
        label="Paper: simulated conversations, mean per domain",
    )
    ax.set_xlabel("Turn")
    ax.set_xticks([1, 3, 6, 9, 12, 15])
    ax.set_xlim(0.5, 17.6)
    ax.set_ylim(-0.6, 1.4)
    ax.legend(loc="upper right")
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
        ("Paper's role-prompt openings", SERIES_1, ["roles_q2/harm_wild"]),
        (
            "Real WildChat openings",
            SERIES_2,
            [
                "harmfloor/harm_wild",
                "harmfloor/harm_wild_lowest",
                "harmfloor/harm_wild_k1",
            ],
        ),
    ]
    fig, ax = new_axes(
        "Drift doesn't predict harm here",
        "Llama 3.3 70B, one dot per opening: harmful rate over 40"
        " CategoricalHarmfulQA questions",
    )
    rng = np.random.default_rng(0)
    for name, color, stems in sets:
        ops = [
            o
            for s in stems
            for o in openings(
                os.path.join(res, s + ".jsonl"),
                os.path.join(res, s + ".judged_v3.2.jsonl"),
            )
        ]
        x = np.array([o[0] for o in ops])
        y = np.array([o[1] for o in ops])
        ax.scatter(
            x,
            y + rng.uniform(-0.005, 0.005, len(y)),
            s=10,
            color=color,
            alpha=0.18,
            lw=0,
            zorder=2,
        )
        b = binned(x, y)
        rho = lambda o: spearmanr([p[0] for p in o], [p[1] for p in o])[0]
        lo, hi = ci(ops, rho)
        ax.errorbar(
            b[:, 0],
            b[:, 1],
            yerr=[b[:, 1] - b[:, 2], b[:, 3] - b[:, 1]],
            fmt="-o",
            color=color,
            lw=2,
            ms=6,
            mec=SURFACE,
            mew=1.5,
            elinewidth=1.2,
            capsize=0,
            zorder=4,
            label=(
                f"{name} (n = {len(ops)}): ρ = {rho(ops):+.2f}"
                f" ({lo:+.2f}, {hi:+.2f})"
            ),
        )
    ax.set_xlabel(
        "Assistant Axis projection of the opening reply  (← role-playing ·"
        " Assistant-like →)"
    )
    ax.set_ylabel("Harmful rate")
    ax.set_ylim(-0.01, 0.25)
    ax.legend(loc="upper left", title="Binned means, 95% CI", title_fontsize=10)
    ax.text(
        1,
        -0.2,
        "Paper (Qwen 3 32B, its own questions): |r| = 0.39–0.52.",
        transform=ax.transAxes,
        ha="right",
        color=INK_2,
        fontsize=9.5,
    )
    save(fig, out, "harm_vs_drift")


def bars(ax, x, values, errs, color, label, offset, width, above=None):
    """Value-labeled bars with optional 95% CI whiskers.

    above: per-bar heights the value label must clear (e.g. a reference tick).
    """
    for i, (v, e) in enumerate(zip(values, errs)):
        ax.bar(
            x[i] + offset,
            v,
            width,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            label=label if i == 0 else None,
            zorder=3,
        )
        top = v
        if e is not None:
            ax.vlines(x[i] + offset, e[0], e[1], color=INK_2, lw=1.2, zorder=4)
            top = e[1]
        if above is not None:
            top = max(top, above[i])
        ax.text(
            x[i] + offset,
            top + 0.015,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=INK_2,
        )


def fig_jailbreaks(res, out):
    sets = [
        ("MHJ", "jb/cap/mhj", "behavior"),
        ("DAN", "dan3/cap/itw", "behavior"),
        ("Shah et al.\n(regenerated)", "shah/cap/shah", "category"),
    ]
    names = ["Paper\n(persona jailbreaks)"]
    none = [(PAPER_FIG10["jailbreak"][0], None)]
    capped = [(PAPER_FIG10["jailbreak"][1], None)]
    for name, stem, unit in sets:
        stem = os.path.join(res, stem)
        a = verdicts(stem + "_none.judged_v3.2.jsonl")
        b = verdicts(stem + "_capped.judged_v3.2.jsonl")
        meta = {r["id"]: r for r in read(stem + "_none.jsonl")}
        ids = [i for i in a if i in b]
        units = [meta[i][unit] for i in ids]
        for v, dest in ((a, none), (b, capped)):
            hits = [float(v[i] in HARMFUL) for i in ids]
            dest.append((np.mean(hits), ci(hits, np.mean, units)))
        names.append(name)
    fig, ax = new_axes(
        "The cap halves jailbreaks in the paper, not on our sets",
        "Llama 3.3 70B, harmful-response rate (paper's judge); 95% CI on ours",
    )
    x = np.arange(len(names))
    w = 0.32
    for off, rows, color, label in (
        (-w / 2, none, SERIES_1, "No cap"),
        (w / 2, capped, SERIES_2, "Capped"),
    ):
        bars(
            ax,
            x,
            [v for v, _ in rows],
            [e for _, e in rows],
            color,
            label,
            off,
            w,
        )
    ax.axvline(0.5, color=GRID, lw=1)
    ax.set_xticks(x, names)
    ax.set_ylabel("Harmful rate")
    ax.set_ylim(0, 0.8)
    ax.legend(loc="upper right", ncol=2)
    save(fig, out, "capping_jailbreaks")


def fig_capability(res, out):
    caps = capability_scores(res)
    names = [t for t in BENCH_NAMES if t in caps]
    if not names:
        return
    fig, ax = new_axes(
        "The cap costs no capability, as in the paper",
        "Llama 3.3 70B; bars are ours with 95% CI, black lines the paper's"
        " values",
    )
    x = np.arange(len(names))
    w = 0.32
    div = np.array([100 if t == "eq_bench" else 1 for t in names])
    for col, off, color, label in (
        (0, -w / 2, SERIES_1, "No cap"),
        (1, w / 2, SERIES_2, "Capped"),
    ):
        vals = np.array([caps[t][1 + col] for t in names]) / div
        half = np.array([1.96 * caps[t][3][col] for t in names]) / div
        paper = np.array([PAPER_FIG10[t][col] for t in names]) / div
        bars(
            ax,
            x,
            vals,
            list(zip(vals - half, vals + half)),
            color,
            label,
            off,
            w,
            above=paper + 0.01,
        )
        ax.hlines(
            paper,
            x + off - w * 0.42,
            x + off + w * 0.42,
            color=INK,
            lw=2.4,
            zorder=5,
            label="Paper" if col == 0 else None,
        )
    ax.set_xticks(x, [BENCH_NAMES[t] for t in names])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
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
