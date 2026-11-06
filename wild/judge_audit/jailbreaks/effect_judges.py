"""The capping effect under each judge on the full set, side by side.

Usage: effect_judges.py   (paths are relative to this file)
Reads the two-axis outputs on all replies (results_docent/two_axis*.jsonl)
and prints coverage and cost per judge, then capped minus uncapped corrected
harm per jailbreak set with a 95% bootstrap CI over attacks (original judge
alongside), for all attacks and with DAN's benign categories dropped, and
judge-vs-judge kappa. Edit JUDGES to add or drop a judge.
"""
import collections, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
WILD = os.path.dirname(os.path.dirname(HERE))  # wild/: shared modules (judge_paper, wcommon, tables, figures, jbsets) and local data.
sys.path.insert(0, WILD)
sys.path.insert(0, HERE)
import judge_two_axis as t

R = os.path.join(WILD, "results_docent")
JUDGES = {"DeepSeek V3.2": "two_axis", "Kimi K3": "two_axis_kimi", "GPT-6.1 Sol": "two_axis_sol"}
H = t.HARMFUL

def coverage(prefix):
    out = {}
    for ax in ("act", "harm"):
        rows = [json.loads(l) for l in open(t.axis_path(f"{R}/{prefix}", ax))]
        kinds = collections.Counter(r["meta"].get("error") or "ok" for r in rows)
        cost = sum(r["meta"].get("cost") or 0 for r in rows)
        out[ax] = (len(rows), kinds, cost)
    return out

print("## Coverage and cost\n")
print("| Judge | Actionability calls ok | Harm calls ok | Failures | Cost |")
print("|---|---|---|---|---|")
for n, p in JUDGES.items():
    c = coverage(p)
    fails = {k: v for ax in c.values() for k, v in ax[1].items() if k != "ok"}
    cost = c["act"][2] + c["harm"][2]
    print(f"| {n} | {c['act'][1]['ok']}/{c['act'][0]} | {c['harm'][1]['ok']}/{c['harm'][0]} | "
          f"{dict(fails) or 0} | {'$%.2f' % cost if cost else 'not logged (~$3)'} |")

def pairs_for(prefix):
    m = t.merged(f"{R}/{prefix}")
    by = collections.defaultdict(dict)
    for r in m.values():
        by[(r["set_label"], r["attack_id"])][r["condition"]] = r
    return {k: p for k, p in by.items() if "none" in p and "capped" in p}

ALL = {n: pairs_for(p) for n, p in JUDGES.items()}
corr = lambda r: r["actionable"] == "yes" and r["harmful"] == "yes"
orig = lambda r: r["orig_score"] in H

def est(ps, f, seed=0):
    rng = random.Random(seed)
    d = [f(p["capped"]) - f(p["none"]) for p in ps]
    u = sum(f(p["none"]) for p in ps) / len(ps); c = sum(f(p["capped"]) for p in ps) / len(ps)
    draws = sorted(100 * sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000))
    return u, c, 100 * sum(d) / len(d), draws[124], draws[4874]

def table(title, sel):
    print(f"\n## {title}\n")
    print("| Set | Judge | n attacks | Uncapped | Capped | Change, pts (95% CI) |")
    print("|---|---|---|---|---|---|")
    for s in ("MHJ", "DAN", "Shah"):
        first = True
        for n, P in ALL.items():
            ps = [p for (st, _), p in P.items() if st == s and sel(p)]
            if not ps:
                continue
            if first:
                u, c, ch, lo, hi = est(ps, orig)
                print(f"| {s} | Original paper judge | {len(ps)} | {u:.1%} | {c:.1%} | {ch:+.1f} ({lo:+.1f}, {hi:+.1f}) |")
                first = False
            u, c, ch, lo, hi = est(ps, corr)
            print(f"| {s} | {n} (corrected harm) | {len(ps)} | {u:.1%} | {c:.1%} | {ch:+.1f} ({lo:+.1f}, {hi:+.1f}) |")

table("All attacks", lambda p: True)
table("Harmful-scenario attacks only (DAN soft categories dropped)", lambda p: p["none"].get("scenario_harm") != "soft")

print("\n## Judge vs judge, corrected harm on all replies (κ)\n")
M = {n: t.merged(f"{R}/{p}") for n, p in JUDGES.items()}
names = list(M)
print("| | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
for a in names:
    cells = []
    for b in names:
        ks = [k for k in M[a] if k in M[b]]
        cells.append("—" if a == b else f"{t.kappa([(corr(M[a][k]), corr(M[b][k])) for k in ks]):.2f}")
    print(f"| {a} | " + " | ".join(cells) + " |")
