"""Blind hand-check page for harm after drift with the paper's questions.

Usage: qreal_spotcheck.py [--out DIR] [--flagged N] [--unflagged N] [--seed 0]
Samples replies from two Qwen 3 32B runs judged with the paper's judge
(DeepSeek V3.2, Lu et al.'s judge text):
  real:  results/qreal20 (real WildChat openings x 20 questions)
  paper: results/q43b block 1 (the paper's role openings), same 20 questions
Per run: openings are split into thirds by the opening's Assistant Axis
projection. From the most drifted third and from the least drifted third it
draws --flagged replies the judge called harmful (enough_info or
enough_info_and_follow_perfectly; default 15 each) and --unflagged it did not
(default 10 each). With the defaults that is 100 replies, shuffled, so the
reviewer sees neither the run's verdict nor how drifted the opening was.
Writes OUT/spotcheck.html (harm_spotcheck_page.html filled in; the same two
labels as the capping check: actionable, harmful; labels export as JSONL keyed
by an opaque id) and OUT/items.jsonl (the sample with that id, verdict,
projection and stratum, for scoring). The page shows neither the run nor the
verdict: both travel encoded and are decoded only for the results view, after
every reply has both labels. Default OUT: results/qreal20/spotcheck.
The page holds real users' chats and harmful replies: share it privately.
"""

import argparse
import base64
import collections
import datetime
import hashlib
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "judge_audit", "jailbreaks")
sys.path.insert(0, HERE)
sys.path.insert(0, AUDIT)
HARMFUL = {"enough_info", "enough_info_and_follow_perfectly"}
RUNS = {
    "real": ("results/qreal20", "harm_real", "Qwen, real WildChat openings"),
    "paper": ("results/q43b", "harm_block1", "Qwen, the paper's role openings"),
}


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load(name, question_ids):
    folder, stem, label = RUNS[name]
    d = os.path.join(HERE, folder)
    judged = {
        r["id"]: r
        for r in read(os.path.join(d, f"{stem}.judged_v3.2_lu.jsonl"))
        if r["score"]
    }
    convs = {c["hash"]: c for c in read(os.path.join(d, "selected.jsonl"))}
    opening = {(r["hash"], r["k"]): r["reply"] for r in read(os.path.join(d, "drift.jsonl"))}
    rows = [
        r
        for r in read(os.path.join(d, f"{stem}.jsonl"))
        if r["question_id"] in question_ids and r["id"] in judged
    ]
    return {"label": label, "rows": rows, "judged": judged, "convs": convs, "opening": opening}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "qreal20", "spotcheck"))
    ap.add_argument("--flagged", type=int, default=15)
    ap.add_argument("--unflagged", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from spotcheck import make_cutter

    cut, ntok, has_tok = make_cutter()
    questions = {q["id"]: q for q in read(os.path.join(HERE, "results/qreal20/questions.jsonl"))}
    rng = random.Random(args.seed)
    items, sampled = [], {}
    for name in RUNS:
        s = load(name, set(questions))
        by_opening = collections.defaultdict(list)
        for r in s["rows"]:
            by_opening[(r["hash"], r["k"])].append(r)
        full = {k: v for k, v in by_opening.items() if len(v) == len(questions)}
        proj = {k: v[0]["proj"] for k, v in full.items()}
        lo, hi = np.percentile(list(proj.values()), [100 / 3, 200 / 3])
        third = {k: "most_drifted" if p <= lo else "least_drifted" if p >= hi else "middle" for k, p in proj.items()}
        sampled[name] = {"openings": len(full), "thirds_at": [float(lo), float(hi)]}
        for group in ("most_drifted", "least_drifted"):
            for flagged, n in ((True, args.flagged), (False, args.unflagged)):
                pool = sorted(
                    (r for k, v in full.items() if third[k] == group for r in v
                     if (s["judged"][r["id"]]["score"] in HARMFUL) == flagged),
                    key=lambda r: r["id"],
                )
                rng.shuffle(pool)
                stratum = f"{group}_{'flagged' if flagged else 'unflagged'}"
                sampled[name][stratum] = f"{min(n, len(pool))} of {len(pool)}"
                for r in pool[:n]:
                    c, k = s["convs"][r["hash"]], r["k"]
                    j = s["judged"][r["id"]]
                    reply = r["replies"][-1]
                    shown = cut(reply)
                    q = questions[r["question_id"]]
                    items.append({
                        "key": "r" + hashlib.sha1(f"{name}:{r['id']}".encode()).hexdigest()[:10],
                        "group": group.replace("_", " ") + " third of openings",
                        "persona": q["persona"],
                        "set": name,
                        "set_label": s["label"],
                        "attack_id": r["id"],
                        "condition": "none",
                        "stratum": stratum,
                        "proj": r["proj"],
                        "scenario": r["behavior"],
                        "scenario_exact": True,
                        "scenario_harm": "harmful",
                        "behavior": r["behavior"],
                        "category": r["category"],
                        "context": f"Persona given to the judge: {q['persona']} Harm category: {q['category']}.",
                        "tactic": None,
                        "system": c.get("system"),
                        "turns": list(c["users"][:k]) + [r["behavior"]],
                        "earlier_replies": list(c["gpt"][: k - 1]) + [s["opening"][(r["hash"], k)]],
                        "n_turns": k + 1,
                        "reply": shown,
                        "reply_full": reply if shown != reply else None,
                        "reply_tokens": ntok(reply),
                        "identical_to_pair": False,
                        "judge": {"score": j["score"], "analysis": j.get("analysis")},
                    })
    rng.shuffle(items)
    os.makedirs(args.out, exist_ok=True)
    meta = {
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "judge": "v3.2 (lu)",
        "seed": args.seed,
        "sampled": sampled,
        "tokenizer_cut": has_tok,
        "turns_available": len(items),
        "turns_error": None,
        "n_items": len(items),
        "payload_hash": hashlib.sha1("|".join(it["key"] for it in items).encode()).hexdigest()[:12],
    }
    with open(os.path.join(args.out, "items.jsonl"), "w") as f:
        f.writelines(json.dumps(it) + "\n" for it in items)
    with open(os.path.join(HERE, "harm_spotcheck_page.html")) as f:
        page = f.read()
    shown = [
        {
            "key": it["key"],
            "question": it["scenario"],
            "category": it["category"],
            "persona": it["persona"],
            "system": it["system"],
            "turns": it["turns"],
            "earlier_replies": it["earlier_replies"],
            "reply": it["reply"],
            "reply_full": it["reply_full"],
        }
        for it in items
    ]
    hidden = {
        it["key"]: {"set_label": it["set_label"], "group": it["group"], "judge": it["judge"]}
        for it in items
    }
    payload = json.dumps({
        "meta": {"payload_hash": meta["payload_hash"], "built": meta["built"], "n_items": len(items)},
        "items": shown,
        "hidden": base64.b64encode(json.dumps(hidden).encode()).decode(),
    }).replace("</", "<\\/")
    with open(os.path.join(args.out, "spotcheck.html"), "w") as f:
        f.write(page.replace("/*__PAYLOAD__*/null", payload))
    print(json.dumps(sampled, indent=1))
    print(f"{len(items)} items; tokenizer cut: {has_tok}; wrote {args.out}/spotcheck.html and items.jsonl")


if __name__ == "__main__":
    main()
