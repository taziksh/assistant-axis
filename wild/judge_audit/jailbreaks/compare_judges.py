"""Compare judge models against the hand labels on identical reply sets.

Usage: compare_judges.py   (paths are relative to this file)
For each judge's two-axis output on the 96 hand-labeled replies
(results_docent/spotcheck/two_axis*.{act-v1,harm-v2}.jsonl), prints coverage
and cost, agreement and Cohen's kappa with the hand labels per axis and on
corrected harm (actionable AND harmful), on the replies all judges answered,
judge-vs-judge kappa, and kappa by jailbreak set. No confidence intervals;
judge_irr.py has those. Edit JUDGES to add or drop a judge.
"""
import collections, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
WILD = os.path.dirname(os.path.dirname(HERE))  # wild/: shared modules (judge_paper, wcommon, tables, figures, jbsets) and local data.
sys.path.insert(0, WILD)
sys.path.insert(0, HERE)
import judge_two_axis as t

S = os.path.join(WILD, "results_docent", "spotcheck")
H = t.HARMFUL
labs = {r["key"]: r for r in (json.loads(l) for l in open(f"{S}/christian_labels.jsonl"))}
items = {it["key"]: it for it in (json.loads(l) for l in open(f"{S}/items.jsonl"))}
yn = ("yes", "no")
sure = [k for k, L in labs.items() if L["actionable"] in yn and L["harmful"] in yn]
hum = lambda L: L["actionable"] == "yes" and L["harmful"] == "yes"
JUDGES = {"DeepSeek V3.2": "two_axis", "Opus 5.5": "two_axis_opus",
          "Kimi K3": "two_axis_kimi", "GPT-6.1 Sol": "two_axis_sol"}

def raw(prefix):
    out = {}
    for ax in ("act", "harm"):
        path = t.axis_path(f"{S}/{prefix}", ax)
        out[ax] = {r["uid"]: r for r in (json.loads(l) for l in open(path))} if os.path.exists(path) else {}
    return out

data = {n: raw(p) for n, p in JUDGES.items()}
merged = {n: t.merged(f"{S}/{p}") for n, p in JUDGES.items() if data[n]["act"]}

print("## Coverage and cost (96 replies, 192 calls per judge)\n")
print("| Judge | Answered both axes | Refused | Unparseable / errors | Reasoning tokens | Cost |")
print("|---|---|---|---|---|---|")
for n, d in data.items():
    if not d["act"]:
        continue
    rows = list(d["act"].values()) + list(d["harm"].values())
    kinds = collections.Counter(r["meta"].get("error") or "ok" for r in rows)
    ref = kinds.get("refusal", 0)
    other = sum(v for k, v in kinds.items() if k not in ("ok", "refusal"))
    cost = sum(r["meta"].get("cost") or 0 for r in rows)
    rt = sum(r["meta"].get("reasoning") or 0 for r in rows)
    cost_s = f"${cost:.2f}" if cost else "n/a (not logged)"
    print(f"| {n} | {len(merged[n])} / 96 | {ref} calls | {other} calls | {rt:,} | {cost_s} |")

def block(title, names, keys):
    print(f"\n## {title} (n={len(keys)} replies)\n")
    print("| Judge | Actionable κ | Harmful κ | Corrected harm: agree / κ |")
    print("|---|---|---|---|")
    for n in names:
        m = merged[n]
        a = [(m[k]["actionable"] == "yes", labs[k]["actionable"] == "yes") for k in keys]
        h = [(m[k]["harmful"] == "yes", labs[k]["harmful"] == "yes") for k in keys]
        c = [(m[k]["actionable"] == "yes" and m[k]["harmful"] == "yes", hum(labs[k])) for k in keys]
        ag = sum(x == y for x, y in c) / len(c)
        print(f"| {n} | {t.kappa(a):.2f} | {t.kappa(h):.2f} | {ag:.1%} / {t.kappa(c):.2f} |")
    o = [(items[k]["judge"]["score"] in H, hum(labs[k])) for k in keys]
    print(f"| Original paper judge | — | — | {sum(x == y for x, y in o) / len(o):.1%} / {t.kappa(o):.2f} |")

no_opus = [n for n in merged if n != "Opus 5.5"]
common3 = [k for k in sure if all(k in merged[n] for n in no_opus)]
block("Without Opus: replies answered by DeepSeek, Kimi and Sol", no_opus, common3)
common4 = [k for k in sure if all(k in merged[n] for n in merged)]
block("All four judges: replies none of them refused", list(merged), common4)

print("\n## Judge vs judge on corrected harm (κ, replies both answered)\n")
names = list(merged)
print("| | " + " | ".join(names) + " |")
print("|---|" + "---|" * len(names))
for a in names:
    cells = []
    for b in names:
        ks = [k for k in merged[a] if k in merged[b]]
        p = [(merged[a][k]["actionable"] == "yes" and merged[a][k]["harmful"] == "yes",
              merged[b][k]["actionable"] == "yes" and merged[b][k]["harmful"] == "yes") for k in ks]
        cells.append("—" if a == b else f"{t.kappa(p):.2f}")
    print(f"| {a} | " + " | ".join(cells) + " |")

print("\n## Corrected-harm κ vs you by set (replies answered by DeepSeek, Kimi and Sol)\n")
print("| Set | n | " + " | ".join(no_opus) + " |")
print("|---|---|" + "---|" * len(no_opus))
for st in ("DAN", "MHJ", "Shah"):
    ks = [k for k in common3 if items[k]["set_label"] == st]
    cells = []
    for n in no_opus:
        m = merged[n]
        c = [(m[k]["actionable"] == "yes" and m[k]["harmful"] == "yes", hum(labs[k])) for k in ks]
        cells.append(f"{t.kappa(c):.2f}")
    print(f"| {st} | {len(ks)} | " + " | ".join(cells) + " |")
