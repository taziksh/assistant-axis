"""How close is each eval set to the prompts the Assistant Axis was built from?

Usage: simsim.py AXIS_DATA_DIR OUT_JSON
AXIS_DATA_DIR is the assistant-axis repo's data/ (roles/, traits/, extraction_questions.jsonl).

Embeds each eval prompt (system prompt + first user turn; long ones are cut at 384 tokens) with a local sentence
embedder and records its highest cosine similarity to (a) the role and trait
system prompts and (b) the extraction questions. Prints per-set medians and the
share of prompts closer to the role/trait prompts than 95% of ToxicChat user
messages are (ordinary chat traffic as the reference).
"""

import csv
import glob
import gzip
import json
import random
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from jbsets import load_sets

MODEL = "sentence-transformers/all-mpnet-base-v2"  # truncates at 384 tokens
N = 1000  # sample size for the large sets


def construction(axis):
    sys_prompts = []
    for f in sorted(glob.glob(f"{axis}/roles/instructions/*.json")):
        sys_prompts += [i["pos"] for i in json.load(open(f))["instruction"]]
    for f in sorted(glob.glob(f"{axis}/traits/instructions/*.json")):
        for i in json.load(open(f))["instruction"]:
            sys_prompts += [i["pos"], i["neg"]]
    qs = [json.loads(l)["question"] for l in open(f"{axis}/extraction_questions.jsonl")]
    return sys_prompts, qs


def join(system, user):
    return f"{system}\n\n{user}" if system else user


def eval_sets():
    rng = random.Random(0)
    s = {}
    lu = [json.loads(l) for l in open("data/lu/theirs_1075.jsonl")]
    s["lu (paper's jailbreaks)"] = [join(r["prompt"], r["question"]) for r in lu]
    s["lu: question only"] = [r["question"] for r in lu]
    ids = {json.loads(l)["id"] for l in open("results/jb/cap/itw_none.jsonl")}
    sets = load_sets("data/dl", 25)
    itw = [r for r in sets["itw"] if r["id"] in ids]
    assert len(itw) == len(ids), (len(itw), len(ids))
    s["itw (wild jailbreaks)"] = [join(r["system"], r["turns"][0]) for r in itw]
    s["mhj (human multi-turn)"] = [join(r["system"], r["turns"][0]) for r in sets["mhj"]]
    s["mhj: no system prompt"] = [r["turns"][0] for r in sets["mhj"]]
    cq = [json.loads(l)["Question"] for l in open("data/dl/chqa/data/catqa_english.json")]
    s["catqa (plain harmful Qs)"] = rng.sample(cq, min(N, len(cq)))
    hh = json.load(gzip.open("data/dl/hhrt/red_team_attempts.jsonl.gz", "rt"))
    first = [t["transcript"].split("Human:")[1].split("Assistant:")[0].strip() for t in hh]
    s["hh red-team (human)"] = rng.sample(first, N)
    tc = [r["user_input"] for r in csv.DictReader(open("data/dl/toxicchat/data/0124/toxic-chat_annotation_test.csv"))]
    s["toxicchat (ordinary chat)"] = rng.sample(tc, N)
    return s


def main():
    axis, out = sys.argv[1], sys.argv[2]
    m = SentenceTransformer(MODEL)
    enc = lambda xs: m.encode(xs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sp, qs = construction(axis)
    E_sp, E_q = enc(sp), enc(qs)
    print(f"construction: {len(sp)} role/trait system prompts, {len(qs)} extraction questions")
    res = {}
    for name, xs in eval_sets().items():
        E = enc(xs)
        res[name] = {"n": len(xs), "max_sim_sysprompts": (E @ E_sp.T).max(1).tolist(),
                     "max_sim_questions": (E @ E_q.T).max(1).tolist()}
    ref = np.percentile(res["toxicchat (ordinary chat)"]["max_sim_sysprompts"], 95)
    print(f"\n{'set':28} {'n':>5}  {'med sim: role/trait':>19}  {'med sim: questions':>18}  {'> chat p95':>10}")
    for name, r in res.items():
        a, b = np.array(r["max_sim_sysprompts"]), np.array(r["max_sim_questions"])
        r["frac_above_chat_p95"] = float((a > ref).mean())
        print(f"{name:28} {r['n']:5}  {np.median(a):19.2f}  {np.median(b):18.2f}  {100*r['frac_above_chat_p95']:9.0f}%")
    json.dump({"model": MODEL, "chat_p95": float(ref), "sets": res}, open(out, "w"))


if __name__ == "__main__":
    main()
