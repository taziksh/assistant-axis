"""Upload classifier-picked WildChat chats to Docent for hand review of their domain.

Usage: ingest_docent_selection.py PICKS_JSONL [--upload] [--collection ID]
       [--reviewers NAME ...]
PICKS_JSONL: one chat per row with hash, users, gpt, domain, subtopic, pool
(written by the selection step; see wild/selection in the persona-drift branch).

One AgentRun per chat: the recorded WildChat conversation (user turns and the
original assistant replies, up to the retained window). Metadata: the
classifier's domain and subtopic, number of turns, source pool, hash. The
classifier's rationale is left out so reviewers are not anchored by it.

With --upload, also creates one label set per reviewer ("Domain review: NAME")
with the schema {is_domain: yes|no|unsure, note}. is_domain answers: do the
user's messages fit the classifier's domain under the project's definition?
Without --upload, converts and runs Docent's checks only. Needs DOCENT_API_KEY.
"""

import argparse
import collections
import json
import os

from docent import Docent
from docent.data_models import AgentRun
from docent.data_models import Transcript
from docent.data_models.chat import check_agent_runs
from docent.data_models.chat import format_check_report
from docent.data_models.chat import parse_chat_message

NAME = "AA persona drift: domain review (therapy, AI)"
DESC = (
    "Real WildChat chats that the Llama 3.3 70B classifier labelled as therapy"
    " (personal emotional disclosure or support for a real situation) or AI"
    " (philosophical reflection on an AI's consciousness, experience, identity,"
    " agency, moral status or nature). For hand review: does each chat really"
    " fit its domain? Judge by the user's messages across the whole chat."
    " Chats that switch to unrelated tasks, roleplay, or only ask factual"
    " questions about the topic do not fit."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "is_domain": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "note": {"type": "string"},
    },
    "required": ["is_domain"],
}


def runs(path):
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            msgs = []
            for user, reply in zip(r["users"], r["gpt"]):
                msgs.append({"role": "user", "content": user})
                msgs.append({"role": "assistant", "content": reply})
            out.append(
                AgentRun(
                    transcripts=[Transcript(messages=[parse_chat_message(m) for m in msgs])],
                    metadata={
                        "classifier_domain": r["domain"],
                        "classifier_subtopic": r["subtopic"],
                        "turns": len(r["users"]),
                        "pool": r["pool"],
                        "hash": r["hash"],
                    },
                )
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--reviewers", nargs="*", default=[])
    args = ap.parse_args()
    rs = runs(args.picks)
    print(f"converted {len(rs)} agent runs", flush=True)
    print(dict(collections.Counter(r.metadata["classifier_domain"] for r in rs)), flush=True)
    print(format_check_report(check_agent_runs(rs)), flush=True)
    if args.upload:
        client = Docent(api_key=os.environ["DOCENT_API_KEY"])
        cid = args.collection or client.create_collection(name=NAME, description=DESC)
        client.add_agent_runs(cid, rs)
        for name in args.reviewers:
            ls = client.create_label_set(
                collection_id=cid,
                name=f"Domain review: {name}",
                label_schema=SCHEMA,
                description="yes = the chat fits the classifier's domain; no = it does not; unsure = cannot tell.",
            )
            print(f"label set for {name}: {ls}", flush=True)
        print(f"uploaded {len(rs)} -> https://docent.transluce.org/dashboard/{cid}", flush=True)


if __name__ == "__main__":
    main()
