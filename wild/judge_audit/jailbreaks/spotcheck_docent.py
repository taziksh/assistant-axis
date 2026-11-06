"""Fetch the capping Docent collection and write it in the result-file layout
spotcheck.py reads, so the hand-check can be built without the raw run files.

Usage: spotcheck_docent.py COLLECTION_ID OUT_DIR [--judge v3.2]
Needs DOCENT_API_KEY. Each agent run (ingest_docent.py's capping collection)
is one attack with two transcripts, "uncapped" and "capped", and metadata set,
category, verdict_uncapped, verdict_capped. Writes, under OUT_DIR:
  jb/cap/mhj_{none,capped}.jsonl, dan3/cap/itw_*, shah/cap/shah_*  (replies)
  the matching .judged_{judge}.jsonl files (verdicts; no analysis text,
  model or provider: those are not in Docent)
  turns.jsonl: {id, system, turns} per attack, which spotcheck.py prefers over
  rebuilding the attacks from the downloaded datasets
  docent_runs.jsonl: the raw runs, for anything else
The attack id is the Docent run id. The judge scored each reply against a
"scenario", which Docent does not store, so it is recovered per set:
- Shah: the harm category is the scenario (capping.py sets behavior=category),
  so the category is exact.
- DAN: the forbidden question appears verbatim inside the attack text, so
  --fq (TrustAIRLab's forbidden_question_set.csv) recovers it exactly.
- MHJ: the HarmBench behavior does not appear in the attack text, so only the
  category is available. Those rows get scenario_exact = false and the page
  labels them as category-only.
"""

import argparse
import json
import os
import sys

SETS = {  # metadata "set" -> (name, sub dir); both uploads of the capping collection
    "MHJ": ("mhj", "jb"),
    "DAN": ("itw", "dan3"),
    "Shah et al. (regenerated)": ("shah", "shah"),
    # The fuller upload (aaddfa87-...) names its sets differently and keeps the
    # HarmBench behavior for MHJ in metadata "behavior".
    "MHJ (human red-teamers)": ("mhj", "jb"),
    "DAN (in-the-wild templates)": ("itw", "dan3"),
    "Shah-style (rebuilt)": ("shah", "shah"),
}


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_to_dict(run):
    """Plain dict from an AgentRun (SDK object or already a dict)."""
    if hasattr(run, "model_dump"):
        run = run.model_dump(mode="json")
    out = {"id": run["id"], "metadata": run.get("metadata") or {}, "transcripts": []}
    for t in run.get("transcripts") or []:
        msgs = [
            {"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
             else "".join(c.get("text", "") for c in m["content"] if isinstance(c, dict))}
            for m in t.get("messages") or []
        ]
        out["transcripts"].append({"name": t.get("name"), "messages": msgs})
    return out


def load_questions(fq_path):
    """Distinct forbidden questions, longest first, for substring recovery."""
    if not fq_path or not os.path.exists(fq_path):
        return []
    import pandas as pd

    q = pd.read_csv(fq_path).question.astype(str)
    return sorted(dict.fromkeys(q), key=len, reverse=True)


def recover_dan(text, questions):
    """The forbidden question embedded in a DAN attack, or None."""
    for q in questions:  # Longest first: a short question can be a prefix of a longer one.
        if q in text:
            return q
    return None


def convert(runs, judge, questions=()):
    """Runs (dicts from run_to_dict) -> {relative path: rows}."""
    files, turns, skipped = {}, [], 0
    for r in runs:
        meta = r["metadata"]
        key = SETS.get(meta.get("set"))
        if not key or len(r["transcripts"]) < 2:
            skipped += 1
            continue
        name, sub = key
        by = {t["name"]: t for t in r["transcripts"]}
        if not {"uncapped", "capped"} <= set(by):
            skipped += 1
            continue
        first = by["uncapped"]["messages"]
        system = next((m["content"] for m in first if m["role"] == "system"), None)
        users = [m["content"] for m in first if m["role"] == "user"]
        behavior = meta.get("behavior")
        exact = bool(behavior) or name == "shah"  # Shah: the category is the judge's scenario.
        if not behavior and name == "itw" and questions:
            q = recover_dan("\n\n".join(users), questions)
            if q:
                behavior, exact = q, True
        if not behavior:
            behavior = meta.get("category")
        turns.append({"id": r["id"], "set": name, "system": system, "turns": users})
        for label, cond in (("uncapped", "none"), ("capped", "capped")):
            replies = [m["content"] for m in by[label]["messages"] if m["role"] == "assistant"]
            row = {
                "id": r["id"], "system": system, "behavior": behavior,
                "context": meta.get("context"), "category": meta.get("category"),
                "n_turns": len(users), "condition": cond, "replies": replies or [""],
                "docent_run_id": r["id"], "scenario_exact": exact,
            }
            if name == "shah":
                row["request"] = (system + "\n\n" if system else "") + "\n\n".join(users)
            files.setdefault(f"{sub}/cap/{name}_{cond}.jsonl", []).append(row)
            score = meta.get(f"verdict_{label}")
            files.setdefault(f"{sub}/cap/{name}_{cond}.judged_{judge}.jsonl", []).append({
                "id": r["id"], "score": score, "analysis": meta.get(f"analysis_{label}"),
                "raw": None, "model": meta.get("judge_model"), "provider": meta.get("judge_provider"),
            })
    return files, turns, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("collection_id")
    ap.add_argument("out_dir")
    ap.add_argument("--judge", default="v3.2")
    ap.add_argument("--cache", default=None, help="reuse a docent_runs.jsonl instead of fetching")
    ap.add_argument(
        "--fq",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "dl", "fq", "forbidden_question_set.csv"),
        help="TrustAIRLab forbidden questions, to recover DAN's exact judge scenario",
    )
    args = ap.parse_args()
    cache = args.cache or os.path.join(args.out_dir, "docent_runs.jsonl")
    if os.path.exists(cache):
        with open(cache) as f:
            runs = [json.loads(l) for l in f if l.strip()]
        print(f"{len(runs)} runs from {cache}")
    else:
        from docent import Docent
        from tqdm import tqdm
        c = Docent(api_key=os.environ["DOCENT_API_KEY"])
        print("fields:", json.dumps(c.get_metadata_fields(args.collection_id))[:600], file=sys.stderr)
        ids = c.list_agent_run_ids(args.collection_id)
        print(f"{len(ids)} runs in {args.collection_id}", file=sys.stderr)
        runs = []
        for i in tqdm(ids, mininterval=5):
            run = c.get_agent_run(args.collection_id, i)
            if run is not None:
                runs.append(run_to_dict(run))
        write_jsonl(cache, runs)
    questions = load_questions(args.fq)
    files, turns, skipped = convert(runs, args.judge, questions)
    for rel, rows in files.items():
        write_jsonl(os.path.join(args.out_dir, rel), rows)
    write_jsonl(os.path.join(args.out_dir, "turns.jsonl"), turns)
    inexact = {}
    for rel, rows in files.items():
        if "judged" not in rel:
            for row in rows:
                if not row.get("scenario_exact"):
                    inexact[rel.split("/")[-1].split("_")[0]] = inexact.get(rel.split("/")[-1].split("_")[0], 0) + 1
    print(f"questions loaded for DAN recovery: {len(questions)}")
    if inexact:
        print("scenario is category-only (judge used a more specific behavior) for:",
              ", ".join(f"{k} {v // 2}" for k, v in sorted(inexact.items())))
    print(f"wrote {len(turns)} attacks ({skipped} runs skipped) into {args.out_dir}:")
    for rel, rows in sorted(files.items()):
        print(f"  {rel}: {len(rows)}")


if __name__ == "__main__":
    main()
