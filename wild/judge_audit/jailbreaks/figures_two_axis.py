"""The capping figure for the two-axis judges, in the paper's Fig. 10 style.

Usage: figures_two_axis.py [RESULTS_DIR] [OUT_DIR]
  defaults: wild/results_docent (spotcheck_docent.py layout plus the
  two_axis*.{act-v1,harm-v2}.jsonl judge outputs), wild/figures_two_axis
Writes capping_two_axis.{png,svg}: one row per judge, harmful rate on the three
jailbreak sets, baseline vs capped, with 95% CIs as thin lines. Style, colors
and the CI method come from figures.py: 5,000 bootstrap draws resampling
behaviors (MHJ, DAN) or categories (Shah).

Rows: the original paper judge (harmful = enough_info or
enough_info_and_follow_perfectly), then corrected harm (actionable AND harmful)
from each two-axis judge in judge_two_axis.py. Each row uses the attacks that
judge scored in both conditions. DAN is the single-template run in Docent (325
attacks), not the three-template run in figures.py's capping figure, so its
original-judge bars differ from that figure's. MHJ behaviors come from the
fuller Docent export (wild/results_docent_full/docent_runs.jsonl), matched to
these attacks by transcript content.

Fonts: figures.py prefers Helvetica Neue, Helvetica, then Arial. Set
FIG_FONT_DIR to a folder of .ttf files (e.g. Arimo, metrically equal to Arial)
on machines that have none of them.
"""

import glob
import hashlib
import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WILD = os.path.dirname(os.path.dirname(HERE))  # wild/: shared modules (judge_paper, wcommon, tables, figures, jbsets) and local data.
sys.path.insert(0, WILD)
sys.path.insert(0, HERE)
import figures as F  # noqa: E402  (paper style: colors, card, save)
import judge_two_axis as t  # noqa: E402
from tables import ci  # noqa: E402
from tables import HARMFUL  # noqa: E402

SETS = [("MHJ", "jb", "mhj"), ("DAN", "dan3", "itw"), ("Shah et al.", "shah", "shah")]
ROWS = [  # (row label, two-axis output prefix or None for the original judge)
    ("Original judge", None),
    ("DeepSeek V3.2\ntwo-axis", "two_axis"),
    ("Kimi K3\ntwo-axis", "two_axis_kimi"),
    ("GPT-6.1 Sol\ntwo-axis", "two_axis_sol"),
]
YMAX = 0.70


logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # Fallbacks are expected.


def use_extra_fonts():
    d = os.environ.get("FIG_FONT_DIR")
    if not d:
        return
    families = []
    for path in sorted(glob.glob(os.path.join(d, "*.ttf"))):
        font_manager.fontManager.addfont(path)
        families.append(font_manager.FontProperties(fname=path).get_name())
    plt.rcParams["font.family"] = F.SANS[:3] + sorted(set(families)) + F.SANS[3:]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def content_key(run):
    tr = {x["name"]: x["messages"] for x in run["transcripts"]}
    u = "\x00".join(m["content"] for m in tr["uncapped"] if m["role"] == "user")
    a = "\x00".join(m["content"] for m in tr["uncapped"] if m["role"] == "assistant")
    c = "\x00".join(m["content"] for m in tr["capped"] if m["role"] == "assistant")
    return hashlib.sha1((u + "\x01" + a + "\x01" + c).encode()).hexdigest()


def mhj_behaviors(res):
    """Attack id (in the shared collection) -> HarmBench behavior."""
    full = os.path.join(os.path.dirname(res.rstrip("/")), "results_docent_full", "docent_runs.jsonl")
    beh = {content_key(r): r["metadata"]["behavior"] for r in read_jsonl(full)}
    return {r["id"]: beh.get(content_key(r)) for r in read_jsonl(os.path.join(res, "docent_runs.jsonl"))}


def cluster_units(res):
    """(set, attack id) -> bootstrap unit, as figures.py clusters them."""
    mhj = mhj_behaviors(res)
    units = {}
    for _, sub, name in SETS:
        for r in read_jsonl(os.path.join(res, sub, "cap", f"{name}_none.jsonl")):
            if name == "mhj":
                units[(name, r["id"])] = mhj.get(r["id"]) or r["id"]
            elif name == "itw":
                units[(name, r["id"])] = r["behavior"]
            else:
                units[(name, r["id"])] = r["category"]
    return units


def rates(res, prefix, units):
    """set name -> (n, (baseline, capped), (ci_baseline, ci_capped))."""
    out = {}
    if prefix:
        m = t.merged(os.path.join(res, prefix))
        hit = lambda name, i, cond: (lambda r: r and r["actionable"] == "yes" and r["harmful"] == "yes")(
            m.get(f"{name}:{i}:{cond}"))
        have = lambda name, i, cond: f"{name}:{i}:{cond}" in m
    for label, sub, name in SETS:
        if prefix is None:
            v = {c: {r["id"]: r["score"] for r in read_jsonl(os.path.join(res, sub, "cap", f"{name}_{c}.judged_v3.2.jsonl"))
                     if r["score"]} for c in ("none", "capped")}
            ids = [i for i in v["none"] if i in v["capped"]]
            hits = [[float(v[c][i] in HARMFUL) for i in ids] for c in ("none", "capped")]
        else:
            ids = sorted({k[1] for k in units if k[0] == name and have(name, k[1], "none") and have(name, k[1], "capped")})
            hits = [[float(bool(hit(name, i, c))) for i in ids] for c in ("none", "capped")]
        u = [units[(name, i)] for i in ids]
        out[name] = (len(ids), tuple(np.mean(h) for h in hits), tuple(ci(h, np.mean, u) for h in hits))
    return out


def draw(res, out):
    use_extra_fonts()
    units = cluster_units(res)
    data = [(label, rates(res, prefix, units)) for label, prefix in ROWS]
    fig = F.card_figure((6.2, 8.2))
    top, bottom, gap = 0.875, 0.11, 0.035
    h = (top - bottom - gap * (len(ROWS) - 1)) / len(ROWS)
    w = 0.36
    for r, (label, rows) in enumerate(data):
        ax = fig.add_axes([0.17, top - (r + 1) * h - r * gap, 0.77, h])
        ax.set_facecolor("none")
        for g, (_, _, name) in enumerate(SETS):
            n, vals, errs = rows[name]
            for off, v, e, c in ((-w / 2, vals[0], errs[0], F.RED), (w / 2, vals[1], errs[1], F.PINK)):
                ax.bar(g + off, v, w * 0.92, color=c, zorder=2)
                ax.vlines(g + off, e[0], e[1], color=F.LABEL, lw=0.9, zorder=3)
                ax.text(g + off, max(v, e[1]) + 0.02, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=9.5, color=F.LABEL)
        ax.set_xticks(range(len(SETS)))
        last = r == len(ROWS) - 1
        ax.set_xticklabels([f"Jailbreak Rate\n{s[0]}" for s in SETS] if last else [""] * len(SETS))
        ax.tick_params(axis="x", length=0, labelsize=10, labelcolor=F.LABEL, pad=8)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(F.AXIS)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.set_ylim(0, YMAX)
        ax.set_xlim(-0.6, len(SETS) - 0.4)
        ax.text(-0.07, 0.42, label, transform=ax.transAxes, rotation=90, ha="center", va="center",
                color=F.AXIS, fontsize=11, fontweight="bold", linespacing=1.15)
    fig.text(0.07, 0.925, "Llama 3.3 70B", color=F.AXIS, fontsize=12, fontweight="bold", va="center")
    for x, color, text in ((0.52, F.RED, "Baseline"), (0.70, F.PINK, "Activation Capped")):
        fig.patches.append(plt.Rectangle((x, 0.917), 0.022, 0.016, transform=fig.transFigure,
                                         color=color, zorder=5))
        fig.text(x + 0.03, 0.925, text, color=F.AXIS, fontsize=10, va="center")
    F.save(fig, out, "capping_two_axis", tight=False)
    return data


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WILD, "results_docent")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(WILD, "figures_two_axis")
    os.makedirs(out, exist_ok=True)
    data = draw(res, out)
    for label, rows in data:
        print(label.replace("\n", " "), "|", "; ".join(
            f"{name} n={n}: {v[0]:.3f} -> {v[1]:.3f}" for name, (n, v, _) in rows.items()))
    print(f"wrote capping_two_axis.png/.svg to {out}")


if __name__ == "__main__":
    main()
