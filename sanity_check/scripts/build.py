"""Collect long English WildChat-1M conversations with GPT's original replies.

Usage: build.py OUT_JSONL [--n N] [--min-turns T] [--max-files M]
Writes one row per conversation: users, replies["gpt"], unusable["gpt"]. Rows are shuffled.
"""
import argparse, random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
from common import BAD_STRINGS, write_jsonl

COLS = ["conversation_hash", "language", "turn", "conversation.list.element.content",
        "conversation.list.element.role"]
fs = HfFileSystem()


def read_file(path, min_turns):
    with fs.open(path) as f:
        t = pq.ParquetFile(f).read(columns=COLS)  # read_table rejects nested column paths.
    lang, turn = t.column("language").to_pylist(), t.column("turn").to_pylist()
    return t.take([i for i, (l, n) in enumerate(zip(lang, turn)) if l == "English" and n >= min_turns]).to_pylist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=16000, help="conversations to keep")
    ap.add_argument("--min-turns", type=int, default=6, help="minimum user turns")
    ap.add_argument("--max-files", type=int, default=None, help="for local tests")
    args = ap.parse_args()

    files = sorted(fs.glob("datasets/allenai/WildChat-1M/data/*.parquet"))[: args.max_files]
    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(lambda p: read_file(p, args.min_turns), files))
    seen, rows, drop = set(), [], Counter()
    for r in (r for part in parts for r in part):
        h, conv = r["conversation_hash"], r["conversation"]
        if h in seen:
            drop["duplicate"] += 1
            continue
        seen.add(h)
        roles = [m["role"] for m in conv]
        texts = [m["content"] for m in conv]
        if len(conv) % 2 or roles != ["user", "assistant"] * (len(conv) // 2):
            drop["roles"] += 1
        elif any(t is None for t in texts):
            drop["none_content"] += 1
        elif any(b in t for t in texts for b in BAD_STRINGS):
            drop["control_strings"] += 1
        else:
            users = texts[::2]
            rows.append({"hash": h, "users": users, "replies": {"gpt": texts[1::2]},
                         "unusable": {"gpt": [False] * len(users)}})
    random.Random(0).shuffle(rows)
    print(f"candidates {len(rows)}; dropped {dict(drop)}; keeping {min(args.n, len(rows))}")
    write_jsonl(args.out, rows[: args.n])


if __name__ == "__main__":
    main()
