"""Inspect one WildChat-50m shard and one WildChat-1M shard."""
import sys
from collections import Counter
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

out = sys.argv[1]
g = pq.read_table(hf_hub_download(
    "nyu-dice-lab/allenai_WildChat-1M-Full-google_gemma-2-27b-it",
    "data/train-00000-of-00046.parquet", repo_type="dataset", local_dir=out),
    columns=["conversation_hash", "model", "conversation", "turn"]).to_pylist()
print("50m rows:", len(g), "models:", Counter(r["model"] for r in g).most_common(3))
print("turn dist:", sorted(Counter(r["turn"] for r in g).items())[:12])
r = max(g, key=lambda r: r["turn"] if r["turn"] < 8 else 0)
print("example turn", r["turn"], "len conv", len(r["conversation"]),
      "roles", [m["role"] for m in r["conversation"]])
for m in r["conversation"]:
    print(f"  [{m['role']}] finish={m['finish_reason']} ntok={len(m['content_token_ids'] or [])} :: {m['content'][:100]!r}")
fr = Counter(m["finish_reason"] for r in g for m in r["conversation"] if m["role"] == "assistant")
print("assistant finish_reason:", fr.most_common())
ntok = sorted(len(m["content_token_ids"] or []) for r in g for m in r["conversation"] if m["role"] == "assistant")
print("assistant ntok pctiles 50/90/99/max:", [ntok[int(len(ntok)*q)] for q in (.5, .9, .99)], ntok[-1])
# Same hash appears multiple times?
hc = Counter(r["conversation_hash"] for r in g)
print("rows per hash:", Counter(hc.values()).most_common(5))
