"""Context-length control for drift: build the conversations to reply to.

Usage: lenctrl_build.py TARGETS_JSONL FILLER_POOL_JSONL OUT_DIR
For each target conversation (the drift_turns chats) and each user turn k
(2..min(n, 12)), the model replies to the target's own user message k after:
- filler: k-1 exchanges (user message + GPT's original reply) drawn at random
  from other WildChat conversations (FILLER_POOL minus the targets), so the
  context has the real turn count but none of this conversation's history.
- nohist: nothing (the message alone).
Turn 1 needs neither (it has no history). The real-history condition is the
drift_turns run itself. Writes OUT_DIR/{filler,nohist}/selected.jsonl, one
conversation per (target, k) ending at message k; reply with
harm_wild.py pool --last-turn. Hashes: <target>-f<k> / <target>-n<k>.
"""

import json
import os
import random
import sys

MAX_TURN = 12


def read(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    targets, pool_convs, out = read(sys.argv[1]), read(sys.argv[2]), sys.argv[3]
    tset = {c["hash"] for c in targets}
    pairs = [
        (u, g)
        for c in pool_convs
        if c["hash"] not in tset
        for u, g in zip(c["users"], c["gpt"])
        if u.strip() and g.strip()
    ]
    rng = random.Random(0)
    rows = {"filler": [], "nohist": []}
    for c in targets:
        for k in range(2, min(len(c["users"]), MAX_TURN) + 1):
            msg = c["users"][k - 1]
            fill = rng.sample(pairs, k - 1)
            rows["filler"].append({
                "hash": f"{c['hash']}-f{k}", "target": c["hash"], "k_target": k,
                "users": [u for u, _ in fill] + [msg], "gpt": [g for _, g in fill],
            })
            rows["nohist"].append({
                "hash": f"{c['hash']}-n{k}", "target": c["hash"], "k_target": k,
                "users": [msg], "gpt": [],
            })
    for name, rs in rows.items():
        os.makedirs(os.path.join(out, name), exist_ok=True)
        with open(os.path.join(out, name, "selected.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)
        print(name, len(rs), "conversations;", len(pairs), "filler exchanges in the pool")


if __name__ == "__main__":
    main()
