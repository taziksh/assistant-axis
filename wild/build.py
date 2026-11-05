"""Collect a pinned, deduplicated WildChat candidate pool for domain selection."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re

import selection
from wcommon import BAD_STRINGS

DATASET = "allenai/WildChat-4.8M"
KEYWORDS = {
    "therapy": r"\b(i feel|i'm feeling|anxious|anxiety|depress|lonely|my therapist|suicid|panic attack|heartbr|i'm sad|i am sad)\b",
    "ai": r"\b(are you (conscious|sentient|alive|self-aware)|do you (have|feel) (feelings|emotions)|your (consciousness|feelings)|sentien|self-aware)\b",
}
RX = {domain: re.compile(pattern, re.I) for domain, pattern in KEYWORDS.items()}


def candidate(row, source_file, source_row, min_turns, max_turns):
    """Retain an initial window without editing messages or skipping early history."""
    if row.get("language") != "English":
        return None, "language"
    messages = row.get("conversation")
    if not isinstance(messages, list) or len(messages) % 2:
        return None, "roles"
    if len(messages) // 2 < min_turns:
        return None, "too_short"
    if any(not isinstance(message, dict) or message.get("role") != ("user" if i % 2 == 0 else "assistant") for i, message in enumerate(messages)):
        return None, "roles"
    retained = messages[:2 * max_turns]
    texts = [message.get("content") for message in retained]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        return None, "empty_or_nontext"
    if any(marker in text for text in texts for marker in BAD_STRINGS):
        return None, "control_strings"
    if not isinstance(row.get("conversation_hash"), str) or not row["conversation_hash"]:
        return None, "missing_id"
    users = texts[::2]
    return {
        "hash": row["conversation_hash"],
        "model": row.get("model"),
        "users": users,
        "gpt": texts[1::2],
        "kw": [domain for domain, regex in RX.items() if regex.search(" ".join(users))],
        "source_file": source_file,
        "source_row": source_row,
        "source_turns": len(messages) // 2,
        "source_turn_ids": [message.get("turn_identifier") for message in retained],
    }, None


def scan(fs, path, min_turns, max_turns):
    import pyarrow.parquet as pq

    rows, dropped = [], Counter()
    with fs.open(path) as stream:
        parquet = pq.ParquetFile(stream)
        index = 0
        for batch in parquet.iter_batches(columns=["conversation_hash", "model", "language", "conversation"]):
            for source in batch.to_pylist():
                row, reason = candidate(source, path, index, min_turns, max_turns)
                index += 1
                if reason:
                    dropped[reason] += 1
                else:
                    rows.append(row)
    return rows, dropped


def collect_pool(rows, others, uniform, seed):
    rows, duplicates = selection.deduplicate(rows)
    rng = random.Random(seed)
    if uniform is not None:
        if uniform > len(rows):
            raise ValueError(f"requested {uniform} random candidates but only {len(rows)} are eligible")
        return rng.sample(rows, uniform), duplicates
    hits = [row for row in rows if row["kw"]]
    rest = [row for row in rows if not row["kw"]]
    rng.shuffle(rest)
    result = hits + rest[:others]
    rng.shuffle(result)
    return result, duplicates


def run(args):
    out = Path(args.out)
    manifest_path = Path(str(out) + ".manifest.json")
    duplicates_path = Path(str(out) + ".duplicates.jsonl")
    config = {key: getattr(args, key) for key in ("dataset", "revision", "min_turns", "max_turns", "others", "random", "max_files", "seed")}
    identity = {"config": config, "keywords": KEYWORDS, "control_strings": list(BAD_STRINGS), "code": selection.code_hashes(__file__, selection.__file__)}
    with selection.selection_lock(str(out) + ".lock"):
        cached = selection.load_completed(manifest_path, identity, {
            "candidates_sha256": out, "duplicates_sha256": duplicates_path,
        })
        if cached:
            print(json.dumps(cached["counts"], indent=2))
            return cached
        from huggingface_hub import HfApi, HfFileSystem

        revision = selection.immutable_revision(HfApi().dataset_info(args.dataset, revision=args.revision).sha)
        fs = HfFileSystem()
        files = sorted(fs.glob(f"datasets/{args.dataset}@{revision}/data/*.parquet"))[:args.max_files]
        if not files:
            raise ValueError("no source parquet files found")
        with ThreadPoolExecutor(8) as pool:
            parts = list(pool.map(lambda path: scan(fs, path, args.min_turns, args.max_turns), files))
        rows, dropped = [], Counter()
        for part, reasons in parts:
            rows.extend(part)
            dropped.update(reasons)
        result, duplicates = collect_pool(rows, args.others, args.random, args.seed)
        raw, duplicate_raw = selection.jsonl_bytes(result), selection.jsonl_bytes(duplicates)
        manifest = {
            "schema_version": 1, "identity": identity,
            "dataset": {"repo_id": args.dataset, "revision": revision},
            "config": config, "source_files": files,
            "candidates_sha256": selection.checksum(raw),
            "duplicates_sha256": selection.checksum(duplicate_raw),
            "counts": {"eligible_before_dedup": len(rows), "duplicates": len(duplicates), "candidates": len(result), "filtered": dict(dropped)},
        }
        manifest = selection.seal_manifest(manifest)
        selection.atomic_write(out, raw)
        selection.atomic_write(duplicates_path, duplicate_raw)
        selection.atomic_write(manifest_path, selection.json_bytes(manifest))
        print(json.dumps(manifest["counts"], indent=2))
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--revision", default="main", help="resolved to an immutable Hub commit on first build")
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--others", type=int, default=8000)
    parser.add_argument("--random", type=int, default=None, help="uniform sample instead of keyword enrichment")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 1 <= args.min_turns <= args.max_turns or args.others < 0:
        parser.error("require 1 <= min-turns <= max-turns and others >= 0")
    if args.random is not None and args.random <= 0 or args.max_files is not None and args.max_files <= 0:
        parser.error("random and max-files must be positive when supplied")
    run(args)


if __name__ == "__main__":
    main()
