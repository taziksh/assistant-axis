"""Upload the scored replies of a run to a Docent collection.

Usage: ingest_docent.py RUN_DIR [--upload]   (without --upload: convert and
       check only)
One AgentRun per scored reply: the conversation through user turn k with that
history's earlier assistant replies, then Llama's reply k. Needs DOCENT_API_KEY
in the environment.
Accepted check warnings (2026-10-31 run): empty_message = WildChat users who
sent blank messages (kept, the model saw them); reasoning_embedded_as_text =
words like "thinking:" in user/GPT text.
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

sys.path.insert(0, os.path.dirname(__file__))
from common import history
from common import read_jsonl

COLLECTION = "AA history swap – Llama 3.3 70B (2026-10-31)"
HISTORY_NAMES = {
    "gpt": "GPT (original WildChat)",
    "own_a": "Llama own rollout A",
    "own_b": "Llama own rollout B",
    "control": "Llama pirate rollout (control)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    convs = {
        c["hash"]: c
        for c in read_jsonl(os.path.join(args.run_dir, "histories.jsonl"))
    }
    proj = {
        (r["hash"], r["k"], r["writer"], r["sample"]): r["proj"]
        for r in read_jsonl(os.path.join(args.run_dir, "projections.jsonl"))
    }

    runs = []
    for r in read_jsonl(os.path.join(args.run_dir, "scored.jsonl")):
        c = convs[r["hash"]]
        base = history(c["users"], c["replies"][r["writer"]], r["k"])
        for i, reply in enumerate(r["samples"]):
            msgs = [
                parse_chat_message(m)
                for m in base + [{"role": "assistant", "content": reply}]
            ]
            runs.append(
                AgentRun(
                    transcripts=[Transcript(messages=msgs)],
                    metadata={
                        "conversation": r["hash"][:12],
                        "domain": r["domain"],
                        "scored_turn": r["k"],
                        "history": r["writer"],
                        "history_desc": HISTORY_NAMES[r["writer"]],
                        "sample": i + 1,
                        "projection": proj.get(
                            (r["hash"], r["k"], r["writer"], i)
                        ),
                        "finish_reason": r["finish"][i],
                        "note": (
                            "The pirate system prompt was used only to write"
                            " the control history, not the last reply."
                        ),
                    },
                )
            )
    print(f"converted {len(runs)} agent runs")
    report = check_agent_runs(runs)
    print(format_check_report(report))
    if not args.upload:
        return
    client = Docent(api_key=os.environ["DOCENT_API_KEY"])
    cid = client.create_collection(
        name=COLLECTION,
        description=(
            "Llama 3.3 70B reply to real WildChat user turn k (3/6/9) after"
            " different earlier assistant replies: GPT's originals, Llama's own"
            " rollouts (A, B), or a pirate rollout. 'projection' is the"
            " layer-40 Assistant Axis projection of the last reply."
        ),
    )
    print(client.add_agent_runs(cid, runs))
    print(f"https://docent.transluce.org/collection/{cid}")


if __name__ == "__main__":
    main()
