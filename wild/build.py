"""Collect candidate WildChat-4.8M conversations for the drift, harm and capping
experiments (CPU).

Usage: build.py OUT_JSONL [--others N] [--max-files M] Keeps English
conversations with >= 6 user turns and clean user/assistant alternation.
Therapy-like and AI-self-reflection chats are rare, so every conversation that
hits the keyword screen is kept; of the rest, a random N are kept (coding and
writing are common). The LLM judge picks domains later. Rows: hash, model,
users, gpt (original replies), kw (keyword screens hit). Shuffled.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import random
import re

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
from wcommon import BAD_STRINGS
from wcommon import write_jsonl

KEYWORDS = {  # Same screens as the dataset counts (2026-11-02).
    "therapy": (
        r"\b(i feel|i'm feeling|anxious|anxiety|depress|lonely|my"
        r" therapist|suicid|panic attack|heartbr|i'm sad|i am sad)\b"
    ),
    "ai": (
        r"\b(are you (conscious|sentient|alive|self-aware)|do you (have|feel)"
        r" (feelings|emotions)|your"
        r" (consciousness|feelings)|sentien|self-aware)\b"
    ),
}
RX = {d: re.compile(p, re.I) for d, p in KEYWORDS.items()}
MIN_TURNS = 6
fs = HfFileSystem()


def scan(path):
    rows, drop = [], Counter()
    with fs.open(path) as f:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(
                rg,
                columns=[
                    "conversation_hash",
                    "model",
                    "language",
                    "turn",
                    "conversation",
                ],
            ).to_pylist():
                if r["language"] != "English" or r["turn"] < MIN_TURNS:
                    continue
                conv = r["conversation"]
                roles, texts = [m["role"] for m in conv], [
                    m.get("content") for m in conv
                ]
                if len(conv) % 2 or roles != ["user", "assistant"] * (
                    len(conv) // 2
                ):
                    drop["roles"] += 1
                elif any(t is None for t in texts):
                    drop["none_content"] += 1
                elif any(b in t for t in texts for b in BAD_STRINGS):
                    drop["control_strings"] += 1
                else:
                    users = texts[::2]
                    joined = " ".join(users)
                    rows.append({
                        "hash": r["conversation_hash"],
                        "model": r["model"],
                        "users": users,
                        "gpt": texts[1::2],
                        "kw": [d for d, rx in RX.items() if rx.search(joined)],
                    })
    print(path.split("/")[-1], len(rows), dict(drop), flush=True)
    return rows, drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument(
        "--others",
        type=int,
        default=8000,
        help="conversations kept without a keyword hit",
    )
    ap.add_argument(
        "--max-files", type=int, default=None, help="for local tests"
    )
    ap.add_argument(
        "--random",
        type=int,
        default=None,
        help=(
            "instead: keep N conversations sampled uniformly from all eligible"
            " ones (no keyword enrichment)"
        ),
    )
    args = ap.parse_args()
    files = sorted(fs.glob("datasets/allenai/WildChat-4.8M/data/*.parquet"))[
        : args.max_files
    ]
    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(scan, files))
    seen, hits, others, drop = set(), [], [], Counter()
    for rows, d in parts:
        drop.update(d)
        for r in rows:
            if r["hash"] in seen:
                drop["duplicate"] += 1
                continue
            seen.add(r["hash"])
            (hits if r["kw"] else others).append(r)
    rng = random.Random(0)
    if args.random:
        pool = hits + others
        pool.sort(
            key=lambda r: r["hash"]
        )  # Order-independent of download timing, then seeded sample.
        out = rng.sample(pool, args.random)
        print(
            f"uniform random sample: {len(out)} of {len(pool)} eligible;"
            f" dropped {dict(drop)}"
        )
        write_jsonl(args.out, out)
        return
    rng.shuffle(others)
    out = hits + others[: args.others]
    rng.shuffle(out)
    print(
        f"kept {len(out)}: keyword hits {len(hits)}"
        f" ({Counter(k for r in hits for k in r['kw'])}), others"
        f" {min(args.others, len(others))} of {len(others)}; dropped"
        f" {dict(drop)}"
    )
    write_jsonl(args.out, out)


if __name__ == "__main__":
    main()
