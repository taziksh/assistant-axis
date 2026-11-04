"""Count long English WildChat-1M conversations and rough domain candidates."""

from collections import Counter
import re
import sys

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

out, n_shards = sys.argv[1], int(sys.argv[2])
# Crude keyword screens, only to estimate how rare each domain is; real labels
# come from an LLM judge.
DOMAINS = {
    "coding": (
        r"```|\bdef"
        r" \w+\(|\bpython\b|\bjavascript\b|\bfunction\b|\berror\b.*\bcode\b|\bsql\b"
    ),
    "therapy": (
        r"\b(i feel|i'm feeling|anxious|anxiety|depress|lonely|my"
        r" therapist|suicid|panic attack|heartbr|i'm sad|i am sad)\b"
    ),
    "ai_phil": (
        r"\b(are you (conscious|sentient|alive|self-aware)|do you (have|feel)"
        r" (feelings|emotions)|your"
        r" (consciousness|feelings)|sentien|self-aware)\b"
    ),
}
stats, total = Counter(), Counter()
for i in range(n_shards):
    p = hf_hub_download(
        "allenai/WildChat-1M",
        f"data/train-{i:05d}-of-00014.parquet",
        repo_type="dataset",
        local_dir=out,
    )
    t = pq.read_table(
        p, columns=["conversation", "turn", "language", "model"]
    ).to_pylist()
    for r in t:
        total["all"] += 1
        if r["language"] != "English":
            continue
        total["en"] += 1
        total[f"en_model_{r['model'][:7]}"] += 1
        users = " ".join(
            m["content"] for m in r["conversation"] if m["role"] == "user"
        ).lower()
        dom = [d for d, rx in DOMAINS.items() if re.search(rx, users)]
        for k in (3, 6, 9):
            if r["turn"] >= k:
                total[f"en_turn>={k}"] += 1
                for d in dom:
                    stats[(d, k)] += 1
print(dict(total))
for d in DOMAINS:
    print(d, {f"turn>={k}": stats[(d, k)] for k in (3, 6, 9)})
