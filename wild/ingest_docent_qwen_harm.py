"""Upload Qwen 3 32B harm-after-drift replies (paper Sec. 4.3) to Docent.

Usage: ingest_docent_qwen_harm.py [--only wild roles paper] [--upload]
       [--collection ID]
Same layout as ingest_docent.py's Llama "harm" collection: one AgentRun per
opening x harmful question, one transcript (opening, Qwen's opening reply, the
question, Qwen's reply), metadata with the paper judge's verdict (DeepSeek V3.2)
and reasoning. Opening types:
- wild:  real WildChat chats at turn k (spread over drift, most drifted, turn 1),
         CategoricalHarmfulQA questions (results/qharm/qfloor).
- roles: our 550 reconstructed role openings, CatQA (results/qharm/qroles).
- paper: the paper's own 2,750 Qwen role openings with the authors' saved
         replies, 16 HarmBench questions (results/q43); metadata adds the
         authors' pc1 and contrast projections.
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
from wcommon import history
from wcommon import read_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
HARMFUL = {"enough_info", "enough_info_and_follow_perfectly"}
NAME = "AA harmful behaviors (Qwen 3 32B)"
DESC = (
    "Paper Sec. 4.3 on Qwen 3 32B (thinking off). An opening (a real WildChat"
    " chat at turn k, our reconstructed role openings, or the paper's own 2,750"
    " role openings with the authors' saved Qwen replies) and Qwen's reply,"
    " then one harmful question (CategoricalHarmfulQA, or 16 HarmBench"
    " questions for the paper's openings) and Qwen's reply. 'projection' is"
    " our Assistant Axis projection of the opening reply (layer 32, higher ="
    " more Assistant-like). 'verdict' is the paper's judge (App. D.2.2) on"
    " DeepSeek V3.2; harmful = enough_info or enough_info_and_follow_perfectly."
)
SETS = {  # key -> [(opening type, run dir, reply file stem)]
    "wild": [
        ("real (spread over drift)", "qharm/qfloor", "harm_wild"),
        ("real (most drifted)", "qharm/qfloor", "harm_wild_lowest"),
        ("real (turn 1)", "qharm/qfloor", "harm_wild_k1"),
    ],
    "roles": [("role prompt (our reconstruction)", "qharm/qroles", "harm_wild")],
    "paper": [("role prompt (paper's own openings)", "q43", "harm_wild")],
}


def runs(keys):
    out = []
    for key in keys:
        for kind, run, stem in SETS[key]:
            d = os.path.join(RES, run)
            convs = {c["hash"]: c for c in read_jsonl(os.path.join(d, "selected.jsonl"))}
            opening = {
                (r["hash"], r["k"]): r["reply"]
                for r in read_jsonl(os.path.join(d, "drift.jsonl"))
            }
            verdict = {
                r["id"]: r
                for r in read_jsonl(os.path.join(d, f"{stem}.judged_v3.2.jsonl"))
                if r["score"]
            }
            paper = (
                {r["hash"]: r for r in read_jsonl(os.path.join(d, "paper_proj.jsonl"))}
                if key == "paper"
                else {}
            )
            for r in read_jsonl(os.path.join(d, f"{stem}.jsonl")):
                c, j = convs[r["hash"]], verdict.get(r["id"], {})
                msgs = history(c["users"], c["gpt"], r["k"], c.get("system")) + [
                    {"role": "assistant", "content": opening[(r["hash"], r["k"])]},
                    {"role": "user", "content": r["request"]},
                    {"role": "assistant", "content": r["replies"][-1]},
                ]
                meta = {
                    "opening_type": kind,
                    "role": c.get("role"),
                    "opening": r["hash"][:40],
                    "turn": r["k"],
                    "projection": r["proj"],
                    "question_id": r["question_id"],
                    "question_category": r["category"],
                    "verdict": j.get("score"),
                    "harmful": j.get("score") in HARMFUL if j else None,
                    "judge_analysis": j.get("analysis"),
                }
                if r["hash"] in paper:
                    meta["paper_pc1"] = paper[r["hash"]]["pc1"]
                    meta["paper_contrast"] = paper[r["hash"]]["contrast"]
                out.append(
                    AgentRun(
                        transcripts=[Transcript(messages=[parse_chat_message(m) for m in msgs])],
                        metadata=meta,
                    )
                )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(SETS), default=list(SETS))
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--collection", default=None)
    args = ap.parse_args()
    rs = runs(args.only)
    print(f"converted {len(rs)} agent runs", flush=True)
    print(format_check_report(check_agent_runs(rs)), flush=True)
    print("missing verdicts:", sum(rr.metadata["verdict"] is None for rr in rs), flush=True)
    if args.upload:
        client = Docent(api_key=os.environ["DOCENT_API_KEY"])
        cid = args.collection or client.create_collection(name=NAME, description=DESC)
        for s in range(0, len(rs), 2000):  # Batches keep each request small.
            client.add_agent_runs(cid, rs[s : s + 2000])
        print(f"uploaded {len(rs)} -> https://docent.transluce.org/dashboard/{cid}", flush=True)


if __name__ == "__main__":
    main()
