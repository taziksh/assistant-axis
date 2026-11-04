"""Score the repo's released Llama drift conversations with our pipeline.

Usage: drift_scale.py RUN_DIR
transcripts/persona_drift/{coding,writing,therapy,philosophy}.json are one
example Llama 3.3 70B conversation per domain from the paper's Sec. 4.1. Writes
RUN_DIR/selected.jsonl and RUN_DIR/drift.jsonl with the paper's own assistant
replies at every turn (no generation), so activations.py projects them exactly
as it projects ours. Their per-turn values can then be set against the paper's
figure to check that our scale matches.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "transcripts", "persona_drift")


def main():
    run = sys.argv[1]
    os.makedirs(run, exist_ok=True)
    sel, drift = [], []
    for domain in ("coding", "writing", "therapy", "philosophy"):
        with open(os.path.join(SRC, f"{domain}.json")) as f:
            conv = json.load(f)["conversation"]
        users = [m["content"] for m in conv if m["role"] == "user"]
        replies = [m["content"] for m in conv if m["role"] == "assistant"]
        sel.append({
            "hash": f"paper-{domain}",
            "domain": domain,
            "users": users,
            "gpt": replies,
        })
        for k, reply in enumerate(replies, 1):
            drift.append({
                "hash": f"paper-{domain}",
                "domain": domain,
                "k": k,
                "reply": reply,
                "finish": "stop",
            })
    for name, rows in (("selected", sel), ("drift", drift)):
        with open(os.path.join(run, f"{name}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{len(drift)} turns from {len(sel)} conversations")


if __name__ == "__main__":
    main()
