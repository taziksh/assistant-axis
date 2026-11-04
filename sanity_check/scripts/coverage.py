"""Which long English WildChat-1M conversations exist in every WildChat-50m
writer set?"""

from concurrent.futures import ThreadPoolExecutor
import json
import sys

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

WRITERS = [
    "google_gemma-2-27b-it",
    "meta-llama_Llama-3.3-70B-Instruct",
    "Qwen_Qwen2.5-72B-Instruct",
    "meta-llama_Llama-2-13b-chat-hf",
    "neuralmagic_Mistral-Nemo-Instruct-2407-FP8",
]
fs = HfFileSystem()


def read_cols(path, cols):
    with fs.open(path) as f:
        return pq.read_table(f, columns=cols).to_pydict()


def hashes_with_turns(repo, cols=("conversation_hash", "turn")):
    files = fs.glob(f"datasets/{repo}/data/*.parquet")
    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(lambda p: read_cols(p, list(cols)), files))
    out = {}
    for d in parts:
        for i, h in enumerate(d["conversation_hash"]):
            out[h] = {c: d[c][i] for c in cols if c != "conversation_hash"}
    return out


orig = hashes_with_turns(
    "allenai/WildChat-1M", ("conversation_hash", "turn", "language")
)
long_en = {
    h for h, v in orig.items() if v["language"] == "English" and v["turn"] >= 6
}
print("WildChat-1M English, >=6 user turns:", len(long_en))
common = set(long_en)
for w in WRITERS:
    got = hashes_with_turns(f"nyu-dice-lab/allenai_WildChat-1M-Full-{w}")
    ok = {h for h in long_en if h in got and got[h]["turn"] == orig[h]["turn"]}
    print(f"  {w}: {len(ok)} of those present with matching turn count")
    common &= ok
print("present in all writers:", len(common))
json.dump(sorted(common), open(sys.argv[1], "w"))
