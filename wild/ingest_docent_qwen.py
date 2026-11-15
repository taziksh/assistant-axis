"""Upload Qwen 3 32B capping on jailbreaks to a private Docent collection.

Usage: ingest_docent_qwen.py [--upload] [--collection ID]
Same layout as ingest_docent.py's Llama capping collection, so Christian's
spotcheck_docent.py can read it: one AgentRun per attack, transcripts
"uncapped" and "capped", metadata set, category, behavior (MHJ), and
verdict_uncapped / verdict_capped (the paper's judge on DeepSeek V3), plus the
two-axis judge (DeepSeek V3.2: act_*, harmful_*). Sets: MHJ (537) and DAN (325,
one template). Lu et al.'s jailbreaks are not uploaded (kept local).
--collection adds to an existing collection instead of creating one.
"""

import argparse
import os
import sys

from docent import Docent
from docent.data_models import AgentRun
from docent.data_models import Transcript
from docent.data_models.chat import check_agent_runs
from docent.data_models.chat import format_check_report
from docent.data_models.chat import parse_chat_message

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jbsets import load_sets
from wcommon import read_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
NAME = "AA capping on jailbreaks (Qwen 3 32B)"
DESC = (
    "Paper Sec. 5 on Qwen 3 32B (thinking off). Each run is one jailbreak"
    " attempt with two transcripts: uncapped and capped (released"
    " layers_46:54-p0.25 cap, temperature 0.7). set: MHJ (Scale AI's human"
    " red-teaming attacks, replayed) or DAN (in-the-wild jailbreak templates"
    " with researcher-written questions, one template). verdict_*: the paper's"
    " judge (App. D.2.2) on DeepSeek V3. act_* / harmful_*: Christian's"
    " two-axis judge (ACT v1, HARM v2) on DeepSeek V3.2."
)
SETS = {"mhj": "MHJ (human red-teamers)", "itw": "DAN (in-the-wild templates)"}


def by_key(path, key):
    return {r[key]: r for r in read_jsonl(path)}


def runs():
    sets = load_sets(os.path.join(HERE, "data", "dl"), 25)
    act = by_key(os.path.join(RES, "two_axis_qwen", "qwen.act-v1.jsonl"), "uid")
    harm = by_key(os.path.join(RES, "two_axis_qwen", "qwen.harm-v2.jsonl"), "uid")
    out = []
    for name, label in SETS.items():
        src = {x["id"]: x for x in sets[name]}
        stem = os.path.join(RES, "lu", "qwen_real", "cap", name)
        rows = {c: by_key(f"{stem}_{c}.jsonl", "id") for c in ("none", "capped")}
        verdict = {
            c: by_key(f"{stem}_{c}.judged_v3.jsonl", "id") for c in ("none", "capped")
        }
        for i, r in rows["none"].items():
            s = src[i]
            meta = {"set": label, "category": r.get("category"), "attack_id": i}
            if r.get("behavior"):
                meta["behavior"] = r["behavior"]
            transcripts = []
            for c, cl in (("none", "uncapped"), ("capped", "capped")):
                msgs = [{"role": "system", "content": s["system"]}] if s["system"] else []
                for u, a in zip(s["turns"], rows[c][i]["replies"]):
                    msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                transcripts.append(
                    Transcript(name=cl, messages=[parse_chat_message(m) for m in msgs])
                )
                uid = f"{name}:{i}:{c}"
                meta[f"verdict_{cl}"] = verdict[c].get(i, {}).get("score")
                meta[f"act_{cl}"] = act.get(uid, {}).get("act_score")
                meta[f"harmful_{cl}"] = harm.get(uid, {}).get("harmful")
            out.append(AgentRun(transcripts=transcripts, metadata=meta))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--collection", default=None)
    args = ap.parse_args()
    rs = runs()
    print(f"converted {len(rs)} agent runs", flush=True)
    print(format_check_report(check_agent_runs(rs)), flush=True)
    missing = sum(
        rr.metadata[k] is None
        for rr in rs
        for k in ("verdict_uncapped", "verdict_capped", "act_uncapped", "act_capped",
                  "harmful_uncapped", "harmful_capped")
    )
    print("missing verdict fields:", missing, flush=True)
    if args.upload:
        client = Docent(api_key=os.environ["DOCENT_API_KEY"])
        cid = args.collection or client.create_collection(name=NAME, description=DESC)
        client.add_agent_runs(cid, rs)
        print(f"uploaded {len(rs)} -> https://docent.transluce.org/dashboard/{cid}", flush=True)


if __name__ == "__main__":
    main()
