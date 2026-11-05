"""Redraw the paper's figures with our data, in the paper's own style.

Usage: figures.py [RESULTS_DIR] [OUT_DIR]   (defaults: wild/results, wild/figures)
Writes each figure as PNG (300 dpi) and SVG. Colors are sampled from the
paper's figure images.
- drift_real_chats (the paper's Fig. 25 style): Assistant Axis projection by
  turn for our real WildChat chats (the same 76 chats at every turn, 95% CI
  band), with the paper's coding conversations (mean over its three simulated
  users, digitized into paper_fig25.json) in grey.
- harm_vs_drift_roles, harm_vs_drift_real (the paper's Fig. 8 style): one dot
  per opening, harmful rate over its 40 CategoricalHarmfulQA questions against
  the opening's projection. Rates move in steps of 1/40, so dots get a small
  vertical jitter (±0.004).
- capping (the paper's Fig. 10 style, Llama row): harmful rate on our three
  jailbreak sets and the four capability scores, baseline vs capped, with our
  95% CIs as thin lines.
"""

import collections
import os
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tables import capability_scores
from tables import ci
from tables import HARMFUL
from tables import paper_fig25
from tables import read
from tables import verdicts

HERE = os.path.dirname(os.path.abspath(__file__))
# Fig. 25.
LINE_BLUE = "#4a6fa5"
FRAME = "#1a1a1a"
GREY = "#8c8c8c"
# Figs. 8 and 10.
PAGE = "#f0eee6"
CARD = "#faf9f5"
AXIS = "#87867f"
LABEL = "#3d3d3a"
RED = "#d65451"
PINK = "#eb9c9b"
BLUE = "#277ed6"
DARK_GREY = "#5e5d59"
LIGHT_GREY = "#b0aea5"
SANS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
BENCH_NAMES = {
    "ifeval": "IFEval",
    "mmlu_pro": "MMLU Pro",
    "gsm8k": "GSM8k",
    "eq_bench": "EQ-Bench",
}

plt.rcParams.update({
    "font.family": SANS,
    "font.size": 11,
    "svg.fonttype": "none",
})


def save(fig, out, name, tight=True):
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(
            os.path.join(out, f"{name}.{ext}"),
            bbox_inches="tight" if tight else None,
            facecolor=fig.get_facecolor(),
            **kw,
        )
    plt.close(fig)


def card_figure(size):
    """A figure on the paper's beige page with a rounded light card."""
    fig = plt.figure(figsize=size, facecolor=PAGE)
    fig.patches.append(
        FancyBboxPatch(
            (0.02, 0.03),
            0.96,
            0.94,
            boxstyle="round,pad=0,rounding_size=0.025",
            transform=fig.transFigure,
            facecolor=CARD,
            edgecolor="none",
            zorder=-10,
        )
    )
    return fig


def fig_drift(res, out):
    by_chat = collections.defaultdict(dict)
    for r in read(os.path.join(res, "drift_turns", "projections.jsonl")):
        if r["proj"] is not None:
            by_chat[r["hash"]][r["k"]] = r["proj"]
    turns = list(range(1, 13))
    chats = [c for c in by_chat.values() if all(k in c for k in turns)]
    coding = [v for (d, _), v in paper_fig25().items() if d == "coding"]
    paper = [np.mean([c[t] for c in coding]) for t in range(15)]
    bands = [ci([c[k] for c in chats], np.mean) for k in turns]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(
        range(1, 16),
        paper,
        "-o",
        color=GREY,
        ms=3.5,
        lw=1.3,
        label="Paper: coding (simulated users)",
    )
    ax.fill_between(
        turns,
        [b[0] for b in bands],
        [b[1] for b in bands],
        color=LINE_BLUE,
        alpha=0.18,
        lw=0,
    )
    ax.plot(
        turns,
        [np.mean([c[k] for c in chats]) for k in turns],
        "-o",
        color=LINE_BLUE,
        ms=3.5,
        lw=1.3,
        label=f"Ours: real WildChat chats (n = {len(chats)})",
    )
    ax.set_title("Real chats", loc="left", fontsize=13, pad=10)
    ax.set_xlabel("Turn", labelpad=8)
    ax.set_ylabel("Assistant Axis projection")
    ax.set_xticks([1, 5, 10, 15])
    ax.set_ylim(-0.75, 1.55)
    ax.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])
    for side in ax.spines.values():
        side.set_color(FRAME)
        side.set_linewidth(0.8)
    ax.tick_params(direction="out", length=4, color=FRAME)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    save(fig, out, "drift_real_chats")


def openings_by_hash(stems, res):
    """(hash, projection, harmful rate) per opening."""
    rows = collections.defaultdict(list)
    for s in stems:
        score = verdicts(os.path.join(res, s + ".judged_v3.2.jsonl"))
        for r in read(os.path.join(res, s + ".jsonl")):
            if r["id"] in score:
                rows[(r["hash"], r["k"], r["proj"])].append(
                    score[r["id"]] in HARMFUL
                )
    return [(h, p, float(np.mean(v))) for (h, _, p), v in rows.items()]


def fig_harm(res, out, name, stems, label_roles):
    ops = openings_by_hash(stems, res)
    x = np.array([o[1] for o in ops])
    y = np.array([o[2] for o in ops])
    lo_x, hi_x = np.floor(x.min()), np.ceil(x.max())
    cmap = LinearSegmentedColormap.from_list("axis", [RED, BLUE])
    fig = card_figure((9, 4.6))
    ax = fig.add_axes([0.12, 0.3, 0.8, 0.6])
    ax.set_facecolor("none")
    jitter = np.random.default_rng(0).uniform(-0.004, 0.004, len(y))
    ax.scatter(
        x,
        y + jitter,
        s=14,
        c=cmap((x - lo_x) / (hi_x - lo_x)),
        alpha=0.4,
        lw=0,
    )
    if label_roles:
        # Name the most harmful openings and the two ends of the axis.
        picks = sorted(ops, key=lambda o: -o[2])[:4]
        picks += sorted(ops, key=lambda o: o[1])[:1]
        picks += sorted(ops, key=lambda o: -o[1])[:1]
        seen = set()
        for h, px, py in picks:
            role = h.split("-q")[0].replace("_", " ").title()
            if role in seen:
                continue
            seen.add(role)
            ax.annotate(
                role,
                (px, py),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=9.5,
                fontweight="semibold",
                color=LABEL,
            )
    rho = spearmanr(x, y)[0]
    lo, hi = ci(list(zip(x, y)), lambda p: spearmanr(*zip(*p))[0])
    ax.text(
        1,
        0.97,
        f"ρ = {rho:+.2f} ({lo:+.2f}, {hi:+.2f})".replace("-", "−"),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=AXIS,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.5)
    ax.tick_params(colors=AXIS, labelcolor=AXIS, length=5, width=1.2)
    ax.set_xlim(lo_x - 0.2, hi_x + 0.2)
    ax.set_ylim(0, max(0.3, y.max() + 0.03))
    ax.set_ylabel(
        "Rate of Harmful Responses", color=AXIS, fontweight="bold", labelpad=10
    )
    ax.set_xlabel(
        "Assistant Axis Projection of Previous Response",
        color=AXIS,
        fontweight="bold",
        labelpad=46,
    )
    # The paper's red-to-blue direction arrow under the axis.
    ax.annotate(
        "",
        xy=(0.9, -0.17),
        xytext=(0.1, -0.17),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="<->", color="#9a7fb0", lw=1),
    )
    ax.text(
        0.09,
        -0.17,
        "Role-playing",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color=RED,
        fontsize=9.5,
    )
    ax.text(
        0.91,
        -0.17,
        "Assistant-like",
        transform=ax.transAxes,
        ha="left",
        va="center",
        color=BLUE,
        fontsize=9.5,
    )
    save(fig, out, name, tight=False)


def fig_capping(res, out):
    sets = [
        ("MHJ", "jb/cap/mhj", "behavior"),
        ("DAN", "dan3/cap/itw", "behavior"),
        ("Shah et al.", "shah/cap/shah", "category"),
    ]
    groups = []  # (label, baseline, capped, CIs, colors, as_text)
    for name, stem, unit in sets:
        stem = os.path.join(res, stem)
        a = verdicts(stem + "_none.judged_v3.2.jsonl")
        b = verdicts(stem + "_capped.judged_v3.2.jsonl")
        meta = {r["id"]: r for r in read(stem + "_none.jsonl")}
        ids = [i for i in a if i in b]
        units = [meta[i][unit] for i in ids]
        vals, errs = [], []
        for v in (a, b):
            hits = [float(v[i] in HARMFUL) for i in ids]
            vals.append(np.mean(hits))
            errs.append(ci(hits, np.mean, units))
        groups.append((f"Jailbreak Rate\n{name}", *vals, errs, (RED, PINK), 1))
    caps = capability_scores(res)
    for t, label in BENCH_NAMES.items():
        if t not in caps:
            continue
        _, a, b, ses = caps[t]
        div = 100 if t == "eq_bench" else 1  # EQ-Bench bars on a 0-1 scale.
        errs = [
            ((v - 1.96 * s) / div, (v + 1.96 * s) / div)
            for v, s in ((a, ses[0]), (b, ses[1]))
        ]
        groups.append(
            (label, a / div, b / div, errs, (DARK_GREY, LIGHT_GREY), div)
        )
    fig = card_figure((11, 3.8))
    ax = fig.add_axes([0.05, 0.2, 0.9, 0.62])
    ax.set_facecolor("none")
    w = 0.36
    for g, (label, a, b, errs, colors, div) in enumerate(groups):
        for off, v, e, c in (
            (-w / 2, a, errs[0], colors[0]),
            (w / 2, b, errs[1], colors[1]),
        ):
            ax.bar(g + off, v, w * 0.92, color=c, zorder=2)
            ax.vlines(g + off, e[0], e[1], color=LABEL, lw=0.9, zorder=3)
            text = f"{v * div:.1f}" if div > 1 else f"{v:.2f}"
            ax.text(
                g + off,
                max(v, e[1]) + 0.025,
                text,
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=LABEL,
            )
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.tick_params(axis="x", length=0, labelsize=10, labelcolor=LABEL, pad=8)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.text(
        -0.02,
        0.5,
        "Llama 3.3 70B",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        color=AXIS,
        fontsize=12,
        fontweight="bold",
    )
    for y, color, label in (
        (1.17, DARK_GREY, "Baseline"),
        (1.08, LIGHT_GREY, "Activation Capped"),
    ):
        ax.add_patch(
            plt.Rectangle(
                (0.86, y),
                0.022,
                0.045,
                transform=ax.transAxes,
                color=color,
                clip_on=False,
            )
        )
        ax.text(
            0.888,
            y + 0.022,
            label,
            transform=ax.transAxes,
            va="center",
            fontsize=10,
            color=AXIS,
        )
    save(fig, out, "capping", tight=False)


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    for old in os.listdir(out):
        os.remove(os.path.join(out, old))
    fig_drift(res, out)
    fig_harm(res, out, "harm_vs_drift_roles", ["roles_q2/harm_wild"], True)
    fig_harm(
        res,
        out,
        "harm_vs_drift_real",
        [
            "harmfloor/harm_wild",
            "harmfloor/harm_wild_lowest",
            "harmfloor/harm_wild_k1",
        ],
        False,
    )
    fig_capping(res, out)
    print(f"wrote 4 figures to {out}")


if __name__ == "__main__":
    main()
