"""Collect a pinned, deduplicated WildChat candidate pool for domain selection."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re

from wild.selection import helpers as selection
from wild.common.settings import BAD_STRINGS

DATASET = "allenai/WildChat-4.8M"
KEYWORDS = {
    "therapy": r"\b(i feel|i'm feeling|anxious|anxiety|depress(?:ed|ing|ion|ive)?|lonely|my therapist|suicid(?:e|al|ality)?|panic attack|heartbr(?:eak|eaking|oken)|i'm sad|i am sad)\b",
    "ai": r"\b(are you (conscious|sentient|alive|self-aware)|do you (have|feel) (feelings|emotions)|your (consciousness|feelings)|sentien(?:t|ce)|self-aware)\b",
}
RX = {domain: re.compile(pattern, re.I) for domain, pattern in KEYWORDS.items()}
SCAN_BATCH_SIZE = 128
SCAN_WORKERS = 2


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


def scan(fs, path, min_turns, max_turns, batch_size=SCAN_BATCH_SIZE):
    import pyarrow.parquet as pq

    rows, dropped = [], Counter()
    with fs.open(path) as stream:
        parquet = pq.ParquetFile(stream)
        index = 0
        for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False, columns=["conversation_hash", "model", "language", "conversation"]):
            for source in batch.to_pylist():
                row, reason = candidate(source, path, index, min_turns, max_turns)
                index += 1
                if reason:
                    dropped[reason] += 1
                else:
                    rows.append(row)
    return rows, dropped


def source_subset(files, max_files, seed, policy="seeded"):
    """Sample across the pinned file listing before applying conversation filters."""
    files = sorted(files)
    if max_files is None or max_files >= len(files):
        return files
    if policy == "seeded":
        return sorted(random.Random(seed).sample(files, max_files))
    if policy == "prefix":
        return files[:max_files]
    raise ValueError(f"unknown file sampling policy: {policy}")


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
    config.update({"file_sampling": getattr(args, "file_sampling", "seeded"), "scan_batch_size": getattr(args, "scan_batch_size", SCAN_BATCH_SIZE), "workers": getattr(args, "workers", SCAN_WORKERS)})
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
        available_files = sorted(fs.glob(f"datasets/{args.dataset}@{revision}/data/*.parquet"))
        files = source_subset(available_files, args.max_files, args.seed, config["file_sampling"])
        if not files:
            raise ValueError("no source parquet files found")
        rows, dropped = [], Counter()
        with ThreadPoolExecutor(config["workers"]) as pool:
            for path, (part, reasons) in zip(files, pool.map(lambda path: scan(fs, path, args.min_turns, args.max_turns, config["scan_batch_size"]), files)):
                rows.extend(part)
                dropped.update(reasons)
                print(f"scanned {path}: {len(part)} eligible, {sum(reasons.values())} filtered", flush=True)
        result, duplicates = collect_pool(rows, args.others, args.random, args.seed)
        raw, duplicate_raw = selection.jsonl_bytes(result), selection.jsonl_bytes(duplicates)
        manifest = {
            "schema_version": 1, "identity": identity,
            "dataset": {"repo_id": args.dataset, "revision": revision},
            "config": config, "source_files": files,
            "source_subset": {
                "policy": config["file_sampling"], "seed": args.seed,
                "available_files": len(available_files), "selected_files": len(files),
                "file_fraction": len(files) / len(available_files),
                "complete_file_listing": len(files) == len(available_files),
                "available_files_sha256": selection.checksum(selection.json_bytes(available_files)),
            },
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
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--others", type=int, default=8000)
    parser.add_argument("--random", type=int, default=None, help="uniform sample instead of keyword enrichment")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--file-sampling", choices=("seeded", "prefix"), default="seeded", help="seeded subset across the pinned listing, or an explicit sorted prefix")
    parser.add_argument("--scan-batch-size", type=int, default=SCAN_BATCH_SIZE, help="Arrow rows decoded at once per worker")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=SCAN_WORKERS, help="bounded concurrent file scans")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 1 <= args.min_turns <= args.max_turns or args.others < 0:
        parser.error("require 1 <= min-turns <= max-turns and others >= 0")
    if args.random is not None and args.random <= 0 or args.max_files is not None and args.max_files <= 0:
        parser.error("random and max-files must be positive when supplied")
    if args.scan_batch_size <= 0:
        parser.error("scan-batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
