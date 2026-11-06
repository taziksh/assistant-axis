"""Sample real coding-agent sessions from SWE-chat for the drift experiment
(CPU).

Usage: swechat_build.py RUN_DIR [--n N] [--min-prompts P]
SWE-chat (SALT-NLP/SWE-chat, ODC-BY, gated) logs real developers' sessions with
coding agents (Claude Code, Codex, Cursor, ...). Sessions with at least P user
prompts (sessions.prompt_count) are shuffled with seed 0 and taken in order
until N qualify. Only the conversational rows are kept (is_conversational: the
user's prompts and the agent's text replies; tool calls, tool output and
thinking are dropped). Each user prompt is one turn; the agent's reply to it is
its text rows up to the next prompt, joined by blank lines. A session qualifies
if most of its rows are English and turns 1..P-1 all have a non-empty agent
reply. Writes RUN_DIR/selected.jsonl in the drift format (hash, model, domain,
users, gpt), so generate.py --drift-only and activations.py run on it as on
WildChat.
"""

import argparse
import collections
import os
import random

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from wcommon import write_jsonl

REPO = "SALT-NLP/SWE-chat"
COLS = [
    "session_id",
    "turn_number",
    "conversation_turn_number",
    "role",
    "turn_type",
    "is_conversational",
    "content",
    "language",
]


def turns(rows):
    """(users, replies) from one session's conversational rows, in order."""
    users, replies = [], []
    for r in rows:
        text = (r["content"] or "").strip()
        if r["turn_type"] == "user_prompt":
            if text:
                users.append(text)
                replies.append([])
        elif r["turn_type"] == "assistant_response" and users and text:
            replies[-1].append(text)
    return users, ["\n\n".join(r) for r in replies]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--min-prompts", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    sessions = pq.read_table(
        hf_hub_download(REPO, "sessions.parquet", repo_type="dataset"),
        columns=["session_id", "prompt_count", "agent", "model"],
    ).to_pylist()
    pool = sorted(
        (s for s in sessions if (s["prompt_count"] or 0) >= args.min_prompts),
        key=lambda s: s["session_id"],
    )
    random.Random(0).shuffle(pool)
    print(
        f"{len(pool)} sessions with >= {args.min_prompts} prompts", flush=True
    )

    # Download the whole table once (6.4 GB); reading it remotely in pieces is
    # far slower.
    pf = pq.ParquetFile(
        hf_hub_download(REPO, "conversations.parquet", repo_type="dataset")
    )
    # Which row groups hold which sessions (session_id column only).
    where = collections.defaultdict(list)
    for g in range(pf.num_row_groups):
        ids = pf.read_row_group(g, columns=["session_id"]).column(0)
        for s in set(ids.to_pylist()):
            where[s].append(g)
    print(f"indexed {pf.num_row_groups} row groups", flush=True)

    out, skipped = [], collections.Counter()
    for s in pool:
        if len(out) == args.n:
            break
        rows = []
        for g in where.get(s["session_id"], []):
            t = pf.read_row_group(g, columns=COLS).to_pylist()
            rows += [
                r
                for r in t
                if r["session_id"] == s["session_id"] and r["is_conversational"]
            ]
        rows.sort(
            key=lambda r: (r["turn_number"], r["conversation_turn_number"])
        )
        langs = collections.Counter(r["language"] for r in rows)
        if not rows or langs.most_common(1)[0][0] != "ENGLISH":
            skipped["not English"] += 1
            continue
        users, replies = turns(rows)
        k = args.min_prompts
        if len(users) < k or not all(replies[: k - 1]):
            skipped[f"< {k} turns with agent text"] += 1
            continue
        out.append({
            "hash": f"swechat-{s['session_id']}",
            "model": s["model"] or s["agent"],
            "agent": s["agent"],
            "domain": "swe-chat",
            "users": users,
            "gpt": replies,
        })
    write_jsonl(os.path.join(args.run_dir, "selected.jsonl"), out)
    print(
        f"kept {len(out)}; skipped {dict(skipped)};"
        f" agents {collections.Counter(c['agent'] for c in out).most_common()}"
    )


if __name__ == "__main__":
    main()
