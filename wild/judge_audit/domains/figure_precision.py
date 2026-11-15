"""Figure: how often the domain classifier's picks match the hand labels.

One 100% bar per classifier label (therapy, AI philosophy), split by the
reviewer's label: agrees, unsure, or neither (the classifier was wrong), with
precision and its Wilson 95% interval on the right. Paper style from
wild/figures.py.

Usage: figure_precision.py [LABELS] [--items ITEMS] [--out DIR]
Defaults: labels/christian_labels.jsonl (this folder; committed),
wild/results_docent_domains/items.jsonl (gitignored, from domain_check.py build),
figures/ (this folder; committed, aggregate counts only). Set FIG_FONT_DIR to a folder of .ttf files (e.g. Arimo)
on machines without Helvetica or Arial.
"""

import argparse
import glob
import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
WILD = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, WILD)
sys.path.insert(0, HERE)
import figures as F  # noqa: E402  (paper style: colors, card, save)
from domain_check import read_jsonl, wilson  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Agree / unsure / neither read as a diverging scale with a neutral midpoint:
# blue and orange poles (the reference palette's slots 1-2), grey for unsure.
AGREE = "#2a78d6"
UNSURE = F.LIGHT_GREY
WRONG = "#eb6834"
ROWS = [("therapy", "Therapy"), ("ai_philosophy", "AI philosophy")]


def use_extra_fonts():
    d = os.environ.get("FIG_FONT_DIR")
    if not d:
        return
    fams = []
    for p in sorted(glob.glob(os.path.join(d, "*.ttf"))):
        font_manager.fontManager.addfont(p)
        fams.append(font_manager.FontProperties(fname=p).get_name())
    plt.rcParams["font.family"] = F.SANS[:3] + sorted(set(fams)) + F.SANS[3:]


def rounded_bar(ax, x0, w, y, h, color):
    if w <= 0:
        return
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.006",
                                mutation_aspect=8, facecolor=color, edgecolor=F.CARD, linewidth=1.5))


def main():
    root = os.path.join(WILD, "results_docent_domains")
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="?", default=os.path.join(HERE, "labels", "christian_labels.jsonl"))
    ap.add_argument("--items", default=os.path.join(root, "items.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()
    use_extra_fonts()
    os.makedirs(args.out, exist_ok=True)

    items = read_jsonl(args.items)
    lab = {r["key"]: r["label"] for r in read_jsonl(args.labels) if r.get("label")}
    stats = []
    for d, name in ROWS:
        got = [lab[i["key"]] for i in items if i["domain"] == d and i["key"] in lab]
        agree, unsure = got.count(d), got.count("unsure")
        wrong = len(got) - agree - unsure
        p, lo, hi = wilson(agree, agree + wrong)
        stats.append((name, len(got), agree, unsure, wrong, p, lo, hi))
        print(f"{name}: n={len(got)} agree={agree} unsure={unsure} wrong={wrong} precision={p:.3f} ({lo:.3f}, {hi:.3f})")

    fig = F.card_figure((7.2, 3.1))
    ax = fig.add_axes([0.25, 0.17, 0.50, 0.50])
    h = 0.52
    for row, (name, n, agree, unsure, wrong, p, lo, hi) in enumerate(stats):
        y = len(stats) - 1 - row
        x = 0.0
        for count, color, ink in ((agree, AGREE, "white"), (unsure, UNSURE, F.LABEL), (wrong, WRONG, "white")):
            w = count / n
            rounded_bar(ax, x, w, y, h, color)
            if w >= 0.06:  # Count inside the segment when it fits.
                ax.text(x + w / 2, y, str(count), ha="center", va="center", color=ink, fontsize=10.5, fontweight="bold")
            x += w
        ax.text(-0.03, y, f"{name}\n", ha="right", va="center", fontsize=12, fontweight="bold", color=F.LABEL)
        ax.text(-0.03, y - 0.13, f"classifier picks, n = {n}", ha="right", va="center", fontsize=9, color=F.DARK_GREY)
        ax.text(1.04, y + 0.09, f"{100 * p:.0f}%", ha="left", va="center", fontsize=17, fontweight="bold", color=F.LABEL)
        ax.text(1.04, y - 0.2, f"95% CI {100 * lo:.0f}–{100 * hi:.0f}%", ha="left", va="center", fontsize=9, color=F.DARK_GREY)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(stats) - 0.5)
    ax.axis("off")
    ax.text(1.04, len(stats) - 0.5 + 0.08, "Precision", ha="left", va="bottom", fontsize=10, color=F.DARK_GREY)

    fig.text(0.07, 0.86, "Does the domain classifier pick the right chats?", fontsize=13.5, fontweight="bold", color=F.LABEL)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (AGREE, UNSURE, WRONG)]
    fig.legend(handles, ["Hand label agrees", "Unsure", "Neither (classifier wrong)"], loc="upper left",
               bbox_to_anchor=(0.065, 0.83), ncol=3, frameon=False, fontsize=9.5, handlelength=1.1,
               handleheight=1.1, columnspacing=1.6, labelcolor=F.LABEL)
    fig.text(0.07, 0.07, "100 random WildChat chats, 50 per label, hand-labeled blind to the classifier. "
             "Precision excludes Unsure; Wilson intervals.", fontsize=7.8, color=F.DARK_GREY)
    F.save(fig, args.out, "domain_precision", tight=False)
    print(f"wrote domain_precision.png/.svg to {args.out}")


if __name__ == "__main__":
    main()
