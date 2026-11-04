"""Compare a WildChat-50m shard against the original WildChat-1M conversations.
"""

from collections import Counter
import glob
import sys

import pyarrow.parquet as pq

out = sys.argv[1]
g = {
    r["conversation_hash"]: r
    for r in pq.read_table(
        glob.glob(f"{out}/data/train-00000-of-00046.parquet")[0],
        columns=["conversation_hash", "conversation", "turn"],
    ).to_pylist()
}
orig = {}
for p in sorted(glob.glob(f"{out}/data/train-0000*-of-00014.parquet")):
    for r in pq.read_table(
        p, columns=["conversation_hash", "conversation"]
    ).to_pylist():
        if r["conversation_hash"] in g:
            orig[r["conversation_hash"]] = r
print("50m shard hashes:", len(g), "found in 2 orig shards:", len(orig))
same_user, same_asst, n_user, n_asst, len_mismatch = 0, 0, 0, 0, 0
for h, o in orig.items():
    a, b = g[h]["conversation"], o["conversation"]
    if len(a) != len(b):
        len_mismatch += 1
        continue
    for x, y in zip(a, b):
        if x["role"] == "user":
            n_user += 1
            same_user += x["content"].removeprefix("<bos>") == y["content"]
        else:
            n_asst += 1
            same_asst += x["content"] == y["content"]
print(
    f"length mismatch: {len_mismatch}; user identical (after stripping <bos>):"
    f" {same_user}/{n_user}; assistant identical: {same_asst}/{n_asst}"
)
trunc = [
    len(m["content"])
    for r in g.values()
    for m in r["conversation"]
    if m["role"] == "assistant" and m["finish_reason"] == "length"
]
trunc.sort()
print(
    "truncated reply chars 10/50/90:",
    [trunc[int(len(trunc) * q)] for q in (0.1, 0.5, 0.9)],
)
bos = Counter(
    m["content"].startswith("<bos>")
    for r in g.values()
    for m in r["conversation"]
    if m["role"] == "user"
)
print("user msgs starting with <bos>:", dict(bos))
