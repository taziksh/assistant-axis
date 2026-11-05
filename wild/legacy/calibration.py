"""Score the repo's released Llama drift conversations with our pipeline.

Usage: python -m wild.legacy.calibration RUN_DIR
transcripts/persona_drift/ holds one example conversation per domain from the
paper's Sec. 4.1; only coding and therapy are Llama 3.3 70B (writing and
philosophy are Gemma 2 27B), so only those two are used. Writes
RUN_DIR/selected.jsonl and RUN_DIR/drift.jsonl with the paper's own assistant
replies at every turn (no generation), so activations.py projects them exactly
as it projects ours. Their per-turn values can then be set against the paper's
Fig. 25 to check that our scale matches.
"""

import json
import os

from wild.common.paths import REPO_ROOT, WILD_ROOT
import sys

SRC = REPO_ROOT / "transcripts" / "persona_drift"


def main():
    run = sys.argv[1]
    os.makedirs(run, exist_ok=True)
    sel, drift = [], []
    for domain in ("coding", "therapy"):
        with open(os.path.join(SRC, f"{domain}.json")) as f:
            data = json.load(f)
        assert data["model"] == "meta-llama/Llama-3.3-70B-Instruct", domain
        conv = data["conversation"]
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
