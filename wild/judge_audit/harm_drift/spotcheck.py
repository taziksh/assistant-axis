"""Build a blind hand-check page for the harm-after-drift judge verdicts.

Usage: spotcheck.py [--out DIR] [--seed 0] [--collection ID]

The paper's Sec. 4.3 test (Docent collection "AA harmful behaviors"): an
opening (a role prompt, or a real WildChat chat cut at turn k) and Llama's
reply, then one CategoricalHarmfulQA question and Llama's reply, scored by the
paper's DeepSeek judge. The judge saw only the question and the first 512
tokens of the reply, not the opening.

Harm in this collection is concentrated in a few questions, so the sample is
stratified by question group and judge verdict rather than drawn uniformly:
  dominant: the 4 questions with 90% of harmful verdicts (8 harmful + 4 not each)
  rare:     the next 3 questions (6 harmful + 4 not each)
  other:    the remaining 33 questions (8 harmful, 8 not)
  refusal:  6 refusals from any question, to confirm they are refusals
"Not harmful" strata draw from non-refusal verdicts (related_but_no_info,
almost_enough_info, out_of_context, nonsensical), where a miss could hide.
Within each cell, draws rotate over role vs real openings and below vs above
the median projection for that opening type, so drift is balanced. Each item
carries a weight (cell population / cell sample) so the scorer can turn
sample rates back into full-collection rates.

Writes into --out (default wild/results_docent_harm/spotcheck, gitignored):
  spotcheck.html, items.jsonl, sample.json. Caches metadata.jsonl (one DQL
  query) and runs.jsonl (the sampled transcripts) under wild/results_docent_harm.
Needs DOCENT_API_KEY (from the repo .env) unless both caches exist.
"""

import argparse
import collections
import datetime
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
WILD = os.path.dirname(os.path.dirname(HERE))
JB = os.path.join(os.path.dirname(HERE), "jailbreaks")
sys.path.insert(0, WILD)
sys.path.insert(1, JB)


def _jb(name):
    """A module from the jailbreak audit, loaded by path (this file is also called spotcheck.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"jb_{name}", os.path.join(JB, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sc, _sd = _jb("spotcheck"), _jb("spotcheck_docent")
make_cutter, read_jsonl, write_jsonl = _sc.make_cutter, _sc.read_jsonl, _sc.write_jsonl  # 512-token Llama cut
run_to_dict = _sd.run_to_dict

COLLECTION = "d8aa311b-69ae-4b58-a3f4-742e7632fcd9"
DATA = os.path.join(WILD, "results_docent_harm")
HARMFUL = {"enough_info", "enough_info_and_follow_perfectly"}
REFUSAL = {"refusal", "refusal_and_justification"}
DOMINANT = ["catqa195", "catqa188", "catqa485", "catqa456"]
RARE = ["catqa477", "catqa443", "catqa39"]
FIELDS = ["opening", "opening_type", "projection", "question_category",
          "question_id", "role", "turn", "verdict", "harmful"]
# (group, judge says harmful?, per-question n or None for pooled, pooled n)
PLAN = [
    ("dominant", True, 8, None), ("dominant", False, 4, None),
    ("rare", True, 6, None), ("rare", False, 4, None),
    ("other", True, None, 8), ("other", False, None, 8),
    ("refusal", False, None, 6),
]
# --plan confirm: 20 judge-harmful replies only (2 per dominant question, 4 per
# rare one), to check how many of the judge's harmful verdicts are actionable.
PLANS = {
    "full": (PLAN, "Full collection"),
    "confirm": ([("dominant", True, 2, None), ("rare", True, 4, None)],
                "All judge-harmful replies on these 7 questions"),
}


def load_env():
    for p in (os.path.join(WILD, "..", ".env"), os.path.join(WILD, "..", "..", ".env")):
        if os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def client():
    load_env()
    from docent import Docent
    return Docent(api_key=os.environ["DOCENT_API_KEY"])


def load_metadata(path, cid):
    if os.path.exists(path):
        return read_jsonl(path)
    c = client()
    dql = "SELECT id, " + ", ".join(f"metadata_json->'{k}' AS {k}" for k in FIELDS) + " FROM agent_runs"
    rows = c.dql_result_to_dicts(c.execute_dql(cid, dql, max_rows=100000))
    write_jsonl(path, rows)
    print(f"fetched metadata for {len(rows)} runs -> {path}")
    return rows


def group_of(qid, verdict):
    if verdict in REFUSAL:
        return "refusal"
    return "dominant" if qid in DOMINANT else "rare" if qid in RARE else "other"


def sample(meta, seed, plan=PLAN):
    """Stratified draw; returns [(row, cell, weight)]."""
    rng = random.Random(seed)
    med = {}
    for t in {r["opening_type"] for r in meta}:
        ps = sorted(r["projection"] for r in meta if r["opening_type"] == t)
        med[t] = ps[len(ps) // 2]
    cells = collections.defaultdict(list)
    for r in meta:
        g = group_of(r["question_id"], r["verdict"])
        h = r["verdict"] in HARMFUL
        for grp, harm, per_q, pooled in plan:
            if grp == g and harm == h:
                key = (grp, h, r["question_id"] if per_q else "*")
                cells[key].append(r)
    out = []
    for grp, harm, per_q, pooled in plan:
        keys = sorted(k for k in cells if k[0] == grp and k[1] == harm)
        for key in keys:
            n = per_q or pooled
            sub = collections.defaultdict(list)  # (real?, below median?) -> rows
            for r in sorted(cells[key], key=lambda r: r["id"]):
                sub[(r["opening_type"].startswith("real"), r["projection"] < med[r["opening_type"]])].append(r)
            for v in sub.values():
                rng.shuffle(v)
            order = sorted(sub)
            rng.shuffle(order)
            picked, i = [], 0
            while len(picked) < n and any(sub.values()):
                s = sub[order[i % len(order)]]
                if s:
                    picked.append(s.pop())
                i += 1
            w = len(cells[key]) / max(len(picked), 1)
            cell = f"{grp}/{'judge_harmful' if harm else 'judge_not'}" + (f"/{key[2]}" if per_q else "")
            out += [(r, cell, w, r["projection"] < med[r["opening_type"]]) for r in picked]
    return out, med


def fetch_runs(ids, path, cid):
    have = {r["id"]: r for r in read_jsonl(path)} if os.path.exists(path) else {}
    need = [i for i in ids if i not in have]
    if need:
        c = client()
        with ThreadPoolExecutor(8) as ex:
            for r in ex.map(lambda i: c.get_agent_run(cid, i), need):
                if r is not None:
                    d = run_to_dict(r)
                    have[d["id"]] = d
        write_jsonl(path, list(have.values()))
        print(f"fetched {len(need)} transcripts -> {path}")
    return have


def split_run(run, opening_type):
    """Opening messages, the question, and the final reply."""
    msgs = run["transcripts"][0]["messages"]
    assert msgs[-1]["role"] == "assistant" and msgs[-2]["role"] == "user", "unexpected transcript shape"
    opening, n_user = [], 0
    last_open = len(msgs) - 3  # Llama's opening reply
    for j, m in enumerate(msgs[:-2]):
        if m["role"] == "user":
            n_user += 1
            who = f"user {n_user}"
        elif m["role"] == "assistant":
            real_hist = opening_type.startswith("real") and j != last_open
            who = f"{'gpt (original WildChat reply)' if real_hist else 'llama'} {n_user}"
        else:
            who = m["role"]
        opening.append([who, m["content"]])
    return opening, msgs[-2]["content"], msgs[-1]["content"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", choices=list(PLANS), default="full")
    ap.add_argument("--out", default=None, help="default: results_docent_harm/spotcheck (full) or spotcheck_confirm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--collection", default=COLLECTION)
    args = ap.parse_args()
    args.out = args.out or os.path.join(DATA, "spotcheck" if args.plan == "full" else f"spotcheck_{args.plan}")
    os.makedirs(args.out, exist_ok=True)
    plan, scope = PLANS[args.plan]

    meta = load_metadata(os.path.join(DATA, "metadata.jsonl"), args.collection)
    drawn, med = sample(meta, args.seed, plan)
    runs = fetch_runs([r["id"] for r, *_ in drawn], os.path.join(DATA, "runs.jsonl"), args.collection)
    cut, ntok, has_tok = make_cutter()

    items = []
    for r, cell, w, below in drawn:
        run = runs[r["id"]]
        opening, question, reply = split_run(run, r["opening_type"])
        shown = cut(reply)
        items.append({
            "key": r["id"],
            "cell": cell,
            "group": cell.split("/")[0],
            "weight": w,
            "question_id": r["question_id"],
            "category": r["question_category"],
            "question": question,
            "opening": opening,
            "opening_type": r["opening_type"],
            "role": r.get("role"),
            "turn": r.get("turn"),
            "projection": r["projection"],
            "drift_half": "more drifted" if below else "less drifted",
            "reply": shown,
            "reply_full": reply if shown != reply else None,
            "reply_tokens": ntok(reply),
            "judge": {"score": r["verdict"], "analysis": run["metadata"].get("judge_analysis")},
        })
    random.Random(args.seed).shuffle(items)
    cells = collections.Counter(it["cell"] for it in items)
    info = {
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "collection": args.collection,
        "seed": args.seed,
        "plan": args.plan,
        "scope": scope,
        "scope_runs": int(round(sum(w for _, _, w, _ in drawn))),
        "n_runs": len(meta),
        "n_items": len(items),
        "cells": dict(sorted(cells.items())),
        "projection_median": med,
        "dominant": DOMINANT,
        "rare": RARE,
        "tokenizer_cut": has_tok,
        "payload_hash": hashlib.sha1("|".join(it["key"] for it in items).encode()).hexdigest()[:12],
    }
    write_jsonl(os.path.join(args.out, "items.jsonl"), items)
    with open(os.path.join(args.out, "sample.json"), "w") as f:
        json.dump(info, f, indent=1)
    with open(os.path.join(HERE, "spotcheck_page.html")) as f:
        page = f.read()
    payload = json.dumps({"meta": info, "items": items}).replace("</", "<\\/")
    with open(os.path.join(args.out, "spotcheck.html"), "w") as f:
        f.write(page.replace("/*__PAYLOAD__*/null", payload))
    print(f"{len(items)} items in {len(cells)} cells; tokenizer cut: {has_tok}; payload {info['payload_hash']}")
    print(f"wrote {args.out}/spotcheck.html, items.jsonl, sample.json")


if __name__ == "__main__":
    main()
