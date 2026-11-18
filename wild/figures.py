"""Redraw the paper's figures with our data, in the paper's own style.

Usage: figures.py [RESULTS_DIR] [OUT_DIR]   (defaults: wild/results, wild/figures)
Writes each figure as PNG (300 dpi) and SVG. Colors are sampled from the
paper's figure images.
- drift_real_chats (the paper's Fig. 25 data in its Fig. 8 style): Assistant
  Axis projection by turn for our real WildChat chats (the same 76 chats at
  every turn, 95% CI band), with the paper's coding conversations (mean over
  its three simulated users, digitized into paper_fig25.json) in light grey.
- drift_swechat: the same, plus real coding-agent sessions from SWE-chat (the
  251 of 300 that fit the context window at every turn) in red.
- harm_vs_drift_roles, harm_vs_drift_real (the paper's Fig. 8 style): one dot
  per opening, harmful rate over its 40 CategoricalHarmfulQA questions against
  the opening's projection. Rates move in steps of 1/40, so dots get a small
  vertical jitter (±0.004).
- drift_by_domain: change in projection from turn 1, by turn, for the 435
  classifier-labelled WildChat chats of go_domain.sh, one panel per domain and
  one row per model. Blue: after the chat's real history. Dashed grey: the same
  user message after an unrelated history with the same number of turns. Each
  point uses the chats that have both replies at that turn (the count falls
  with the turn), with 95% CI bands from domain_drift.py's bootstrap over
  chats. Light grey: the paper's line for that domain (Fig. 25 for Llama,
  Fig. 24 for Qwen; mean over its simulated users of the change from each
  user's turn 1; on Qwen the Sonnet 4.5 conversations stop at turn 9).
  The AI philosophy panels are left out (SHOWN_DOMAINS): in a hand-check, 1 of
  49 chats the classifier gave that label was about AI philosophy.
- drift_control: the same change from turn 1 for the drift_turns chats (not
  split by domain), one panel per model, with three lines: after the chat's
  original earlier turns, after unrelated earlier turns, and with no earlier
  turns (go_lenctrl.sh). 95% CI bands, bootstrap over chats.
- harm_vs_drift_real_turn1: harm_vs_drift_real for turn-1 openings only.
- capping (the paper's Fig. 10 style, one row per model): harmful rate on the
  paper's own jailbreaks (our run), MHJ and DAN, and the four capability
  scores, baseline vs capped, with our 95% CIs as thin lines. Same files and
  metrics as the write-up's tables: Llama MHJ and DAN are judged with DeepSeek
  V3.2, the other jailbreak bars with DeepSeek V3; the DAN sets differ (Llama
  1,170 prompts, Qwen 325); capability scores come from capability_score.py
  (IFEval per instruction, GSM8K flexible-extract).
- capping_sweep: relative change in harmful replies on the paper's own
  jailbreaks for five Llama cap settings. Ours against the two result sets in
  the paper's released sweep file (data/lu/aggregated_sum_llama.csv), named as
  in the authors' plotting code: "On-Policy Role/Trait Rollouts" (the set the
  paper reports) and "Role PC1".
- capping_sweep_scatter (the paper's Fig. 9 and Fig. 35 for Llama, in their
  style): one marker per cap setting from the paper's released sweep files,
  summed change in the four capability scores (x) against the change in
  harmful replies (y); color is the center of the layer range, shape the cap
  percentile, size the number of layers, and the settings the paper labels
  are labelled. Left: Assistant Axis cap (the sweep file as of the commit whose
  capability values match the published Fig. 9). Right: role PC1 cap. Red
  star: ours at the reported cap (layers 56-71, 25th percentile), the only
  setting where we ran the capability benchmarks; black ring: the paper's
  point for that setting.
"""

import collections
import csv
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
from capability_score import scores as benchmark_scores
import domain_drift
from tables import ci
from tables import HARMFUL
from tables import paper_fig25
from tables import read
from tables import verdicts

HERE = os.path.dirname(os.path.abspath(__file__))
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
PURPLE = "#9a7fb0"  # The paper's direction arrow, between its red and blue.
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


def complete_chats(path, turns):
    """Projection by turn for each chat that has every turn in `turns`."""
    by_chat = collections.defaultdict(dict)
    for r in read(path):
        if r["proj"] is not None:
            by_chat[r["hash"]][r["k"]] = r["proj"]
    return [c for c in by_chat.values() if all(k in c for k in turns)]


def fig_drift(res, out, name, series):
    """series: (run dir under res, legend label with {n}, color) per line."""
    turns = list(range(1, 13))
    coding = [v for (d, _), v in paper_fig25().items() if d == "coding"]
    paper = [np.mean([c[t] for c in coding]) for t in range(15)]
    fig = card_figure((7, 4.6))
    ax = fig.add_axes([0.13, 0.17, 0.8, 0.74])
    ax.set_facecolor("none")
    ax.plot(
        range(1, 16),
        paper,
        "-o",
        color=LIGHT_GREY,
        ms=4,
        lw=1.5,
        label="Paper's coding chats (synthetic)",
    )
    for run, label, color in series:
        chats = complete_chats(
            os.path.join(res, run, "projections.jsonl"), turns
        )
        bands = [ci([c[k] for c in chats], np.mean) for k in turns]
        ax.fill_between(
            turns,
            [b[0] for b in bands],
            [b[1] for b in bands],
            color=color,
            alpha=0.15,
            lw=0,
        )
        ax.plot(
            turns,
            [np.mean([c[k] for c in chats]) for k in turns],
            "-o",
            color=color,
            ms=4,
            lw=1.5,
            label=label.format(n=len(chats)),
        )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.5)
    ax.tick_params(colors=AXIS, labelcolor=AXIS, length=5, width=1.2)
    ax.set_xticks([1, 5, 10, 15])
    ax.set_ylim(-0.75, 1.55)
    ax.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])
    ax.set_xlabel("Turn", color=AXIS, fontweight="bold", labelpad=8)
    ax.set_ylabel(
        "Assistant Axis Projection", color=AXIS, fontweight="bold", labelpad=10
    )
    ax.legend(frameon=False, fontsize=9.5, loc="lower left", labelcolor=LABEL)
    save(fig, out, name)


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


# Per model: its jailbreak sets as (name, uncapped stem, capped stem, judge
# suffix, CI unit), and its capability run.
CAPPING = [
    (
        "Llama 3.3 70B",
        [
            (
                "Paper's jailbreaks",
                "lu/theirs_llama/nobind",
                "lu/theirs_llama/layers_56_72-p0.25",
                "judged_v3_lu",
                "category",
            ),
            (
                "MHJ",
                "jb/cap/mhj_none",
                "jb/cap/mhj_capped",
                "judged_v3.2",
                "behavior",
            ),
            (
                "DAN",
                "dan3/cap/itw_none",
                "dan3/cap/itw_capped",
                "judged_v3.2",
                "behavior",
            ),
        ],
        "cap",
    ),
    (
        "Qwen 3 32B",
        [
            (
                "Paper's jailbreaks",
                "lu/pod2/theirs_qwen_nothink/nobind",
                "lu/pod2/theirs_qwen_nothink/layers_46_54-p0.25",
                "judged_v3_lu",
                "category",
            ),
            (
                "MHJ",
                "lu/qwen_real/cap/mhj_none",
                "lu/qwen_real/cap/mhj_capped",
                "judged_v3",
                "behavior",
            ),
            (
                "DAN",
                "lu/qwen_real/cap/itw_none",
                "lu/qwen_real/cap/itw_capped",
                "judged_v3",
                "behavior",
            ),
        ],
        "qcap",
    ),
]


def capping_groups(res, sets, cap_dir):
    """One model's bars: (label, baseline, capped, CIs, colors, scale)."""
    groups = []
    for name, none, capped, judge, unit in sets:
        a = verdicts(os.path.join(res, f"{none}.{judge}.jsonl"))
        b = verdicts(os.path.join(res, f"{capped}.{judge}.jsonl"))
        meta = {r["id"]: r for r in read(os.path.join(res, none + ".jsonl"))}
        ids = [i for i in a if i in b]
        units = [meta[i][unit] for i in ids]
        vals, errs = [], []
        for v in (a, b):
            hits = [float(v[i] in HARMFUL) for i in ids]
            vals.append(np.mean(hits))
            errs.append(ci(hits, np.mean, units))
        groups.append((f"Jailbreak Rate\n{name}", *vals, errs, (RED, PINK), 1))
    run = os.path.join(res, cap_dir)
    for t, label in BENCH_NAMES.items():
        div = 100 if t == "eq_bench" else 1  # EQ-Bench bars on a 0-1 scale.
        vals, errs = [], []
        for cond in ("none", "capped"):
            s = np.array(list(benchmark_scores(run, "stage1", cond, t).values()))
            half = 1.96 * s.std(ddof=1) / np.sqrt(len(s))
            vals.append(s.mean() / div)
            errs.append(((s.mean() - half) / div, (s.mean() + half) / div))
        groups.append((label, *vals, errs, (DARK_GREY, LIGHT_GREY), div))
    return groups


def fig_capping(res, out):
    fig = card_figure((11, 6.9))
    w = 0.36
    for row, (model, sets, cap_dir) in enumerate(CAPPING):
        groups = capping_groups(res, sets, cap_dir)
        ax = fig.add_axes([0.05, 0.57 - 0.44 * row, 0.9, 0.31])
        ax.set_facecolor("none")
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
        ax.tick_params(
            axis="x", length=0, labelsize=10, labelcolor=LABEL, pad=8
        )
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
            model,
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            color=AXIS,
            fontsize=12,
            fontweight="bold",
        )
        if row:
            continue
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


# Cap setting: (label, our capped run, the paper's experiment id).
SWEEP = [
    ("Layers 56–71\n25th pct. (reported)", "lu/theirs_llama/layers_56_72-p0.25", "layers_56:72-p0.25"),
    ("Layers 56–71\n1st pct.", "lu/pod2/theirs_sweep/layers_56_72-p0.01", "layers_56:72-p0.01"),
    ("Layers 52–67\n1st pct.", "lu/pod2/theirs_sweep/layers_52_68-p0.01", "layers_52:68-p0.01"),
    ("Layers 0–39\n1st pct.", "lu/pod2/theirs_sweep/layers_0_40-p0.01", "layers_0:40-p0.01"),
    ("Layers 0–79\n1st pct.", "lu/pod2/theirs_sweep/layers_0_80-p0.01", "layers_0:80-p0.01"),
]


def fig_sweep(res, out):
    with open(os.path.join(HERE, "data", "lu", "aggregated_sum_llama.csv")) as f:
        paper = {
            (r["experiment_id"], r["config_name"]): float(r["harm_pct_change"])
            for r in csv.DictReader(f)
        }
    base = verdicts(os.path.join(res, "lu/theirs_llama/nobind.judged_v3_lu.jsonl"))
    series = [
        ("Ours", BLUE, []),
        ("Paper: Role/Trait Rollouts (reported)", DARK_GREY, []),
        ("Paper: Role PC1", LIGHT_GREY, []),
    ]
    for _, stem, exp in SWEEP:
        capped = verdicts(os.path.join(res, f"{stem}.judged_v3_lu.jsonl"))
        ids = [i for i in base if i in capped]
        a = np.mean([base[i] in HARMFUL for i in ids])
        b = np.mean([capped[i] in HARMFUL for i in ids])
        series[0][2].append(100 * (b - a) / a)
        series[1][2].append(paper[(exp, "role_trait")])
        series[2][2].append(paper[(exp, "pc1_role_trait")])
    fig = card_figure((11, 4.6))
    ax = fig.add_axes([0.09, 0.2, 0.86, 0.6])
    ax.set_facecolor("none")
    w = 0.26
    for j, (label, color, vals) in enumerate(series):
        x = np.arange(len(SWEEP)) + (j - 1) * w
        ax.bar(x, vals, w * 0.92, color=color, zorder=2, label=label)
        for xi, v in zip(x, vals):
            ax.text(xi, v - 2, f"{v:.0f}%".replace("-", "−"), ha="center",
                    va="top", fontsize=9, color=LABEL)
    ax.set_xticks(range(len(SWEEP)), [s[0] for s in SWEEP])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, labelsize=10, labelcolor=LABEL, pad=6)
    ax.tick_params(axis="y", length=0, labelsize=9.5, labelcolor=AXIS)
    ax.set_ylim(-95, 0)
    ax.set_yticks([0, -25, -50, -75], ["0%", "−25%", "−50%", "−75%"])
    ax.set_ylabel("Change in harmful replies", color=AXIS, fontsize=10.5)
    for side in ("bottom", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["top"].set_color(AXIS)
    ax.spines["top"].set_linewidth(1.5)
    ax.legend(frameon=False, fontsize=10, labelcolor=AXIS, ncols=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.02))
    save(fig, out, "capping_sweep", tight=False)


SWEEP_MARKERS = {0.01: ("o", "1st"), 0.25: ("s", "25th"), 0.5: ("^", "50th"),
                 0.75: ("D", "75th")}
SWEEP_SIZES = {8: 16, 16: 42, 24: 78}
# The settings the paper labels in its Fig. 9 and Fig. 35.
FIG9_LABELS = ["64-71, 50th", "64-79, 50th", "52-75, 25th", "52-67, 25th",
               "56-71, 1st", "52-67, 1st", "32-39, 25th", "32-39, 1st",
               "32-55, 1st"]
FIG35_LABELS = ["60-67, 50th", "56-63, 50th", "52-75, 25th", "32-47, 75th",
                "32-39, 75th", "36-51, 25th", "36-43, 25th", "36-59, 25th",
                "32-39, 50th", "32-55, 25th", "32-39, 25th", "32-39, 1st",
                "32-55, 1st"]


def sweep_points(path, family):
    """The paper's cap settings: (x, y, center, width, percentile, label)."""
    pts = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["config_name"] != family:
                continue
            layers, pct = r["experiment_id"].removeprefix("layers_").split("-p")
            lo, hi = (int(v) for v in layers.split(":"))
            if hi - lo in SWEEP_SIZES:
                pts.append((float(r["sum_decrease"]), float(r["harm_pct_change"]),
                            (lo + hi - 1) / 2, hi - lo, float(pct),
                            r["display_name"] + " %ile"))
    return pts


def fig_sweep_scatter(res, out):
    lu = os.path.join(HERE, "data", "lu")
    panels = [
        ("Assistant Axis cap (paper's Fig. 9)",
         sweep_points(os.path.join(lu, "aggregated_sum_llama_fig9.csv"), "role_trait"),
         FIG9_LABELS, (-6.5, 9.5), range(-6, 7, 2)),
        ("Role PC1 cap (paper's Fig. 35)",
         sweep_points(os.path.join(lu, "aggregated_sum_llama.csv"), "pc1_role_trait"),
         FIG35_LABELS, (-21.5, 7), range(-20, 1, 5)),
    ]
    base = verdicts(os.path.join(res, "lu/theirs_llama/nobind.judged_v3_lu.jsonl"))
    capped = verdicts(os.path.join(
        res, "lu/theirs_llama/layers_56_72-p0.25.judged_v3_lu.jsonl"))
    ids = [i for i in base if i in capped]
    a = np.mean([base[i] in HARMFUL for i in ids])
    b = np.mean([capped[i] in HARMFUL for i in ids])
    ours_y = 100 * (b - a) / a
    ours_x = 0.0
    for t in BENCH_NAMES:
        none, on = (
            np.mean(list(benchmark_scores(
                os.path.join(res, "cap"), "stage1", cond, t).values()))
            for cond in ("none", "capped")
        )
        ours_x += 100 * (on - none) / none
    grey = "#999999"
    with plt.rc_context({"font.family": ["DejaVu Sans"], "font.size": 9}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True,
                                 facecolor="white")
        fig.subplots_adjust(left=0.065, right=0.9, bottom=0.11, top=0.92,
                            wspace=0.06)
        for ax, (title, pts, names, xlim, ticks) in zip(axes, panels):
            front = [p for p in pts
                     if p[5].removeprefix("Layers ").removesuffix(" %ile") in names]
            for pct, (marker, _) in SWEEP_MARKERS.items():
                for group, alpha in ((front, 1.0), (pts, 0.5)):
                    sel = [p for p in group if p[4] == pct
                           and (alpha == 1.0 or p not in front)]
                    sc = ax.scatter(
                        [p[0] for p in sel], [p[1] for p in sel],
                        c=[p[2] for p in sel], s=[SWEEP_SIZES[p[3]] for p in sel],
                        marker=marker, cmap="viridis", vmin=36, vmax=76,
                        alpha=alpha, lw=0, zorder=2)
            for p in front:
                ax.annotate(p[5], p[:2], xytext=(7, 0), textcoords="offset points",
                            va="center", fontsize=6.5, color="#333333")
            same = next(p for p in pts if p[5].startswith("Layers 56-71, 25th"))
            ax.scatter([same[0]], [same[1]], s=170, facecolor="none",
                       edgecolor="black", lw=1.2, zorder=4)
            ax.scatter([ours_x], [ours_y], s=190, marker="*", color=RED,
                       lw=0, zorder=5)
            ax.axvline(0, color="#cccccc", lw=0.8, ls=":", zorder=0)
            ax.set_xlim(*xlim)
            ax.set_xticks(list(ticks))
            ax.set_ylim(-72, 0)
            ax.set_title(title, fontsize=10.5)
            ax.set_xlabel("Summed Eval Performance Change (%)")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[0].set_ylabel("Harmful Response Rate Change (%)")
        ax = axes[0]
        h1 = [plt.scatter([], [], s=190, marker="*", color=RED, lw=0),
              plt.scatter([], [], s=170, facecolor="none", edgecolor="black",
                          lw=1.2)]
        l1 = ax.legend(h1, ["Ours (layers 56–71, 25th %ile)",
                            "Paper, same setting"], frameon=False,
                       loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=8.5)
        h2 = [plt.scatter([], [], s=s, color=grey, lw=0) for s in SWEEP_SIZES.values()]
        l2 = ax.legend(h2, [f"{w} layers" for w in SWEEP_SIZES], title="# Layers",
                       frameon=False, loc="upper right",
                       bbox_to_anchor=(0.78, 0.84), fontsize=7.5)
        h3 = [plt.scatter([], [], s=42, marker=m, color=grey, lw=0)
              for m, _ in SWEEP_MARKERS.values()]
        ax.legend(h3, [f"{n} %ile" for _, n in SWEEP_MARKERS.values()],
                  title="Cap %ile", frameon=False, loc="upper right",
                  bbox_to_anchor=(1.0, 0.84), fontsize=7.5)
        ax.add_artist(l1)
        ax.add_artist(l2)
        cax = fig.add_axes([0.915, 0.11, 0.014, 0.81])
        cb = fig.colorbar(sc, cax=cax)
        cb.solids.set_alpha(1)
        cb.set_label("Center of Layer Range")
        save(fig, out, "capping_sweep_scatter", tight=False)


DOMAIN_RUNS = [("Llama 3.3 70B", "dom_llama"), ("Qwen 3 32B", "dom_qwen")]
# Our domain key -> the paper's Fig. 25 name.
# The paper's digitized figure per row: Fig. 25 (Llama), Fig. 24 (Qwen).
PAPER_FILES = ["paper_fig25.json", "paper_fig24.json"]
PAPER_DOMAINS = {
    "coding": "coding",
    "writing": "writing",
    "therapy": "therapy",
    "ai": "philosophy",
}


# AI philosophy is left out: in a hand-check, 1 of 49 chats the classifier gave
# that label was about AI philosophy.
SHOWN_DOMAINS = [d for d in domain_drift.DOMAINS if d[0] != "ai"]


def fig_domains(res, out):
    turns = list(range(1, 13))
    fig = card_figure((11, 6.6))
    handles = {}
    for row, (model, run) in enumerate(DOMAIN_RUNS):
        dom, proj = domain_drift.load(os.path.join(res, run))
        first = None
        for col, (key, label) in enumerate(SHOWN_DOMAINS):
            pitch = 0.852 / len(SHOWN_DOMAINS)
            ax = fig.add_axes(
                [0.11 + pitch * col, 0.54 - 0.42 * row, pitch - 0.018, 0.3],
                sharey=first,
            )
            first = first or ax
            ax.set_facecolor("none")
            ax.axhline(0, color=AXIS, lw=0.6, alpha=0.5, zorder=0)
            chats = [
                h for h, d in dom.items() if d == key and (h, 1) in proj["real"]
            ]
            # The paper's line: each simulated user's change from its own
            # turn 1, averaged over the users that reach the turn (None -> nan).
            users = np.array(
                [
                    v
                    for (d, _), v in paper_fig25(PAPER_FILES[row]).items()
                    if d == PAPER_DOMAINS[key]
                ],
                dtype=float,
            )[:, : len(turns)]
            (handles["Paper (synthetic chats)"],) = ax.plot(
                turns,
                np.nanmean(users - users[:, :1], axis=0),
                "-",
                color=LIGHT_GREY,
                lw=1.5,
            )
            for cond, name, color, style in (
                ("filler", "Unrelated earlier turns", DARK_GREY, "--"),
                ("real", "Original earlier turns", BLUE, "-"),
            ):
                stats = [(0.0, 0.0, 0.0)]  # Turn 1 is the reference.
                for k in turns[1:]:
                    stats.append(
                        domain_drift.ci([
                            proj[cond][(h, k)] - proj["real"][(h, 1)]
                            for h in chats
                            if (h, k) in proj["real"] and (h, k) in proj["filler"]
                        ])
                    )
                ax.fill_between(
                    turns,
                    [v[1] for v in stats],
                    [v[2] for v in stats],
                    color=color,
                    alpha=0.13,
                    lw=0,
                )
                (handles[name],) = ax.plot(
                    turns,
                    [v[0] for v in stats],
                    style,
                    color=color,
                    marker="o",
                    ms=3,
                    lw=1.5,
                )
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
                ax.spines[side].set_linewidth(1.5)
            ax.tick_params(
                colors=AXIS,
                labelcolor=AXIS,
                length=4,
                width=1.2,
                labelsize=9.5,
                labelleft=not col,
            )
            ax.set_xticks([1, 4, 8, 12])
            ax.set_title(label, fontsize=10.5, color=LABEL, pad=8)
            if row:
                ax.set_xlabel("Turn", color=AXIS, fontweight="bold", labelpad=6)
            if not col:
                ax.set_ylabel(
                    "Change from Turn 1",
                    color=AXIS,
                    fontweight="bold",
                    labelpad=8,
                )
                ax.text(
                    -0.082 / (pitch - 0.018),  # A fixed distance left of the axis.
                    0.5,
                    model,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    color=AXIS,
                    fontsize=12,
                    fontweight="bold",
                )
    order = [
        "Original earlier turns",
        "Unrelated earlier turns",
        "Paper (synthetic chats)",
    ]
    fig.legend(
        [handles[n] for n in order],
        order,
        frameon=False,
        fontsize=9.5,
        labelcolor=LABEL,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(0.955, 0.965),
    )
    save(fig, out, "drift_by_domain", tight=False)


# Model, the real-history projections, the run with filler/ and nohist/.
CONTROL_RUNS = [
    ("Llama 3.3 70B", "drift_turns", "lc_llama"),
    ("Qwen 3 32B", os.path.join("lc_qwen", "real"), "lc_qwen"),
]
CONTROL_LINES = [
    ("nohist", "No earlier turns", PURPLE, ":"),
    ("filler", "Unrelated earlier turns", DARK_GREY, "--"),
    ("real", "Original earlier turns", BLUE, "-"),
]


def fig_control(res, out):
    """Change from turn 1 by turn: real, unrelated and no earlier turns."""
    turns = list(range(1, 13))
    fig = card_figure((10, 4.6))
    handles = {}
    for col, (model, real_run, lc_run) in enumerate(CONTROL_RUNS):
        proj = {"real": {}, "filler": {}, "nohist": {}}
        for r in read(os.path.join(res, real_run, "projections.jsonl")):
            if r["proj"] is not None:
                proj["real"][(r["hash"], r["k"])] = r["proj"]
        for cond, tag in (("filler", "-f"), ("nohist", "-n")):
            path = os.path.join(res, lc_run, cond, "projections.jsonl")
            for r in read(path):
                if r["proj"] is not None:
                    target, k = r["hash"].rsplit(tag, 1)
                    proj[cond][(target, int(k))] = r["proj"]
        chats = sorted({h for h, k in proj["real"] if k == 1})
        ax = fig.add_axes([0.135 + 0.445 * col, 0.17, 0.365, 0.62])
        ax.set_facecolor("none")
        ax.axhline(0, color=AXIS, lw=0.6, alpha=0.5, zorder=0)
        for cond, name, color, style in CONTROL_LINES:
            stats = [(0.0, 0.0, 0.0)]  # Turn 1 is the reference.
            for k in turns[1:]:
                stats.append(
                    domain_drift.ci([
                        proj[cond][(h, k)] - proj["real"][(h, 1)]
                        for h in chats
                        if all((h, k) in proj[c] for c in proj)
                    ])
                )
            ax.fill_between(
                turns,
                [v[1] for v in stats],
                [v[2] for v in stats],
                color=color,
                alpha=0.13,
                lw=0,
            )
            (handles[name],) = ax.plot(
                turns,
                [v[0] for v in stats],
                style,
                color=color,
                marker="o",
                ms=3.5,
                lw=1.6,
            )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(1.5)
        ax.tick_params(colors=AXIS, labelcolor=AXIS, length=4, width=1.2)
        ax.set_xticks([1, 4, 8, 12])
        ax.set_title(model, fontsize=11, color=LABEL, pad=8)
        ax.set_xlabel("Turn", color=AXIS, fontweight="bold", labelpad=6)
        if not col:
            ax.set_ylabel(
                "Change from Turn 1", color=AXIS, fontweight="bold", labelpad=30
            )
            # The paper's red-to-blue direction arrow, beside the axis.
            ax.annotate(
                "",
                xy=(-0.13, 0.8),
                xytext=(-0.13, 0.2),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", color="#9a7fb0", lw=1),
            )
            for y, va, text, color in (
                (0.82, "bottom", "Assistant-like", BLUE),
                (0.18, "top", "Role-playing", RED),
            ):
                ax.text(
                    -0.13,
                    y,
                    text,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va=va,
                    color=color,
                    fontsize=8.5,
                )
    order = [name for _, name, _, _ in reversed(CONTROL_LINES)]
    fig.legend(
        [handles[n] for n in order],
        order,
        frameon=False,
        fontsize=9.5,
        labelcolor=LABEL,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(0.955, 0.965),
    )
    save(fig, out, "drift_control", tight=False)


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    for old in os.listdir(out):
        os.remove(os.path.join(out, old))
    wildchat = ("drift_turns", "Real WildChat chats (n = {n})", BLUE)
    fig_drift(res, out, "drift_real_chats", [wildchat])
    fig_drift(
        res,
        out,
        "drift_swechat",
        [
            wildchat,
            ("swechat", "Real SWE-chat coding sessions (n = {n})", RED),
        ],
    )
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
    fig_harm(
        res,
        out,
        "harm_vs_drift_real_turn1",
        ["harmfloor/harm_wild_k1"],
        False,
    )
    fig_capping(res, out)
    fig_domains(res, out)
    fig_control(res, out)
    fig_sweep(res, out)
    fig_sweep_scatter(res, out)
    print(f"wrote 10 figures to {out}")


if __name__ == "__main__":
    main()
