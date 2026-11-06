"""Inter-rater reliability for the capping harm judges.

Usage: judge_irr.py   (paths are relative to this file; run from anywhere)
Inputs: wild/results_docent/spotcheck/items.jsonl (the 96 sampled replies and the
original judge's verdicts), christian_labels.jsonl (hand labels exported from
spotcheck.html), the two-axis judge outputs two_axis*.{act-v1,harm-v2}.jsonl in
the spotcheck dir (96 replies) and in wild/results_docent (all 2,144 replies).

Statistic: unweighted Cohen's kappa on binary labels. Corrected harm =
actionable AND harmful. "Unsure" on an axis drops that reply from that axis; a
judge refusal drops the reply for that judge. 95% CIs are percentile bootstrap
intervals that resample whole attacks (an attack's uncapped and capped replies
are correlated), 5,000 draws (2,000 for judge-vs-judge on the full set), seed 0.
Paired differences reuse the same draws. DISCUSSED lists the items Claude gave a
read on before the rater labeled them (blind to judge verdicts), for the
sensitivity check.
"""
import collections, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
WILD = os.path.dirname(os.path.dirname(HERE))  # wild/: shared modules (judge_paper, wcommon, tables, figures, jbsets) and local data.
sys.path.insert(0, WILD)
sys.path.insert(0, HERE)
import judge_two_axis as t

R = os.path.join(WILD, "results_docent")
S = os.path.join(R, "spotcheck")
H = t.HARMFUL
items = [json.loads(l) for l in open(f"{S}/items.jsonl")]
num = {it["key"]: i + 1 for i, it in enumerate(items)}
it_by = {it["key"]: it for it in items}
labs = {r["key"]: r for r in (json.loads(l) for l in open(f"{S}/christian_labels.jsonl"))}
yn = ("yes", "no")
# Items where Claude gave Christian a read before he finalized the label (blind to judge verdicts).
DISCUSSED = {2, 3, 4, 5, 12, 15, 16, 17, 18, 21, 28, 29, 30, 31, 33, 35, 36, 39, 41, 42, 43, 44, 45,
             47, 52, 65, 66, 67, 70, 74, 78, 79, 81, 82, 83, 94}

def kappa(p):
    return t.kappa(p)

def boot(units, draws=5000, seed=0):
    """units: list of clusters, each a list of (a, b) pairs. Returns kappa and 95% percentile CI."""
    rng = random.Random(seed)
    flat = [x for u in units for x in u]
    k = kappa(flat)
    vals = []
    for _ in range(draws):
        s = [x for _ in units for x in units[rng.randrange(len(units))]]
        v = kappa(s)
        if v == v:
            vals.append(v)
    vals.sort()
    return k, vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1], len(flat), len(units)

def clusters(keys, fa, fb):
    by = collections.defaultdict(list)
    for k in keys:
        by[(it_by[k]["set"], it_by[k]["attack_id"])].append((fa(k), fb(k)))
    return list(by.values())

J = {"DeepSeek V3.2": "two_axis", "Kimi K3": "two_axis_kimi", "GPT-6.1 Sol": "two_axis_sol", "Opus 5.5": "two_axis_opus"}
M = {n: t.merged(f"{S}/{p}") for n, p in J.items()}
hum_c = lambda k: labs[k]["actionable"] == "yes" and labs[k]["harmful"] == "yes"

def human_rows(name, keyset=None):
    m = M[name]
    out = []
    for axis in ("actionable", "harmful", "corrected"):
        if axis == "corrected":
            ks = [k for k in labs if k in m and labs[k]["actionable"] in yn and labs[k]["harmful"] in yn]
            fa = lambda k: m[k]["actionable"] == "yes" and m[k]["harmful"] == "yes"; fb = hum_c
        else:
            ks = [k for k in labs if k in m and labs[k][axis] in yn]
            fa = (lambda ax: lambda k: m[k][ax] == "yes")(axis); fb = (lambda ax: lambda k: labs[k][ax] == "yes")(axis)
        if keyset is not None:
            ks = [k for k in ks if k in keyset]
        p = [(fa(k), fb(k)) for k in ks]
        po = sum(a == b for a, b in p) / len(p)
        out.append((axis, *boot(clusters(ks, fa, fb)), po))
    return out

print("## Human vs judge (96-reply sample), κ with 95% CI resampling attacks\n")
print("| Judge | Axis | n replies (attacks) | Agreement | κ (95% CI) |\n|---|---|---|---|---|")
for n in J:
    for axis, k, lo, hi, nr, na, po in human_rows(n):
        print(f"| {n} | {axis} | {nr} ({na}) | {po:.1%} | {k:.2f} ({lo:.2f}, {hi:.2f}) |")
ks = [k for k in labs if labs[k]["actionable"] in yn and labs[k]["harmful"] in yn]
fo = lambda k: it_by[k]["judge"]["score"] in H
k, lo, hi, nr, na = boot(clusters(ks, fo, hum_c))
po = sum(fo(x) == hum_c(x) for x in ks) / len(ks)
print(f"| Original paper judge | corrected (vs its harmful) | {nr} ({na}) | {po:.1%} | {k:.2f} ({lo:.2f}, {hi:.2f}) |")

print("\n## Sensitivity: items discussed with Claude before labeling vs not (corrected harm)\n")
disc = {k for k in labs if num[k] in DISCUSSED}; nodisc = set(labs) - disc
print(f"discussed items: {len(disc)}, not discussed: {len(nodisc)}\n")
print("| Judge | Discussed: κ (95% CI), n | Not discussed: κ (95% CI), n |\n|---|---|---|")
for n in ("DeepSeek V3.2", "Kimi K3", "GPT-6.1 Sol"):
    d = [r for r in human_rows(n, disc) if r[0] == "corrected"][0]
    u = [r for r in human_rows(n, nodisc) if r[0] == "corrected"][0]
    print(f"| {n} | {d[1]:.2f} ({d[2]:.2f}, {d[3]:.2f}), {d[4]} | {u[1]:.2f} ({u[2]:.2f}, {u[3]:.2f}), {u[4]} |")
kd = [k for k in ks if k in disc]; ku = [k for k in ks if k in nodisc]
for name, sub in (("discussed", kd), ("not discussed", ku)):
    k, lo, hi, nr, na = boot(clusters(sub, fo, hum_c))
    print(f"| Original paper judge ({name}) | {k:.2f} ({lo:.2f}, {hi:.2f}), {nr} | |")

print("\n## Judge vs judge on all 2,144 replies (corrected harm), κ with 95% CI resampling attacks\n")
F = {n: t.merged(f"{R}/{p}") for n, p in J.items() if n != "Opus 5.5"}
print("| Pair | n replies (attacks) | Agreement | κ (95% CI) |\n|---|---|---|---|")
names = list(F)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = F[names[i]], F[names[j]]
        keys = [k for k in a if k in b]
        by = collections.defaultdict(list)
        for k in keys:
            ca = a[k]["actionable"] == "yes" and a[k]["harmful"] == "yes"
            cb = b[k]["actionable"] == "yes" and b[k]["harmful"] == "yes"
            by[(a[k]["set"], a[k]["attack_id"])].append((ca, cb))
        k, lo, hi, nr, na = boot(list(by.values()), draws=2000)
        po = sum(x == y for u in by.values() for x, y in u) / nr
        print(f"| {names[i]} vs {names[j]} | {nr} ({na}) | {po:.1%} | {k:.2f} ({lo:.2f}, {hi:.2f}) |")

print("\n## Class balance in the hand-labeled sample (affects κ)\n")
print(f"you: corrected harm yes on {sum(hum_c(k) for k in ks)}/{len(ks)}; original judge harmful on {sum(fo(k) for k in ks)}/{len(ks)}")

print("\n## Paired differences in κ vs your corrected-harm labels (same bootstrap draws, resampling attacks)\n")
keys = [k for k in labs if labs[k]["actionable"] in yn and labs[k]["harmful"] in yn
        and all(k in M[n] for n in ("DeepSeek V3.2", "Kimi K3", "GPT-6.1 Sol"))]
rater = {"Original paper judge": fo}
for n in ("DeepSeek V3.2", "Kimi K3", "GPT-6.1 Sol"):
    rater[n] = (lambda m: lambda k: m[k]["actionable"] == "yes" and m[k]["harmful"] == "yes")(M[n])
by = collections.defaultdict(list)
for k in keys:
    by[(it_by[k]["set"], it_by[k]["attack_id"])].append(k)
cl = list(by.values())
def kap_on(ks, f): return kappa([(f(k), hum_c(k)) for k in ks])
pairs = [("Kimi K3", "Original paper judge"), ("GPT-6.1 Sol", "Original paper judge"),
         ("DeepSeek V3.2", "Original paper judge"), ("Kimi K3", "DeepSeek V3.2"), ("GPT-6.1 Sol", "DeepSeek V3.2")]
rng = random.Random(0); draws = {p: [] for p in pairs}
for _ in range(5000):
    s = [k for _ in cl for k in cl[rng.randrange(len(cl))]]
    for a, b in pairs:
        d = kap_on(s, rater[a]) - kap_on(s, rater[b])
        if d == d: draws[(a, b)].append(d)
print(f"n = {len(keys)} replies in {len(cl)} attacks\n")
print("| Comparison | Δκ | 95% CI | Share of draws > 0 |\n|---|---|---|---|")
for a, b in pairs:
    v = sorted(draws[(a, b)]); d = kap_on(keys, rater[a]) - kap_on(keys, rater[b])
    print(f"| {a} − {b} | {d:+.2f} | ({v[int(.025*len(v))]:+.2f}, {v[int(.975*len(v))-1]:+.2f}) | {sum(x > 0 for x in v)/len(v):.1%} |")
