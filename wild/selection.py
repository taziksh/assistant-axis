"""CPU-only helpers for versioned, auditable WildChat selection."""

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile



def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def file_checksum(path):
    return checksum(Path(path).read_bytes())


def read_jsonl(path):
    # Split on LF, not Unicode line separators that may occur in chat text.
    return [json.loads(line) for line in Path(path).read_bytes().split(b"\n") if line.strip()]


def jsonl_bytes(rows):
    return b"".join(json_bytes(row) for row in rows)


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def selection_lock(path):
    """Prevent concurrent builders or judges from publishing mixed artifacts."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another selection process is writing these outputs") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def immutable_revision(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("an immutable 40-character Hub revision is required")
    return value


def code_hashes(*paths):
    return {Path(path).name: file_checksum(path) for path in paths}


def seal_manifest(manifest):
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return {**payload, "manifest_sha256": checksum(json_bytes(payload))}


def verify_manifest(manifest):
    if manifest.get("manifest_sha256") != seal_manifest(manifest)["manifest_sha256"]:
        raise ValueError("selection manifest fingerprint mismatch")


def load_completed(manifest_path, identity, outputs):
    """Return a verified cache, or reject stale, legacy, and partial outputs."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        if any(Path(path).exists() for path in outputs.values()):
            raise ValueError("outputs exist without a completion manifest; use a new output location")
        return None
    manifest = json.loads(manifest_path.read_bytes())
    verify_manifest(manifest)
    if manifest.get("schema_version") != 1 or manifest.get("identity") != identity:
        raise ValueError("selection inputs, policy, revisions, or settings changed; use a new output location")
    for key, path in outputs.items():
        if not Path(path).exists() or file_checksum(path) != manifest.get(key):
            raise ValueError(f"incomplete or modified selection output: {path}")
    return manifest




def deduplicate(rows):
    """Keep longest snapshots first; ties use ID and content, never scan order.

    Prefix matching requires the entire shorter transcript, not a shared greeting.
    Shared source-turn IDs detect branches when source metadata is available.
    """
    rows = sorted(rows, key=lambda row: (-len(row["users"]), row["hash"], checksum(json_bytes(row))))
    seen_ids, prefixes, source_turns = {}, {}, {}
    kept, excluded = [], []
    for row in rows:
        pairs = list(zip(row["users"], row["gpt"]))
        full = checksum(json_bytes(pairs))
        turn_ids = {json.dumps(x, sort_keys=True) for x in row.get("source_turn_ids", []) if x is not None}
        duplicate_of = seen_ids.get(row["hash"])
        reason = "duplicate_id"
        if duplicate_of is None:
            duplicate_of = prefixes.get(full)
            reason = "duplicate_transcript_or_prefix"
        if duplicate_of is None:
            duplicate_of = next((source_turns[key] for key in sorted(turn_ids) if key in source_turns), None)
            reason = "shared_source_turn"
        if duplicate_of is not None:
            excluded.append({"hash": row["hash"], "reason": reason, "duplicate_of": duplicate_of})
            continue
        kept.append(row)
        seen_ids[row["hash"]] = row["hash"]
        for n in range(1, len(pairs) + 1):
            prefixes[checksum(json_bytes(pairs[:n]))] = row["hash"]
        for key in turn_ids:
            source_turns[key] = row["hash"]
    return sorted(kept, key=lambda row: row["hash"]), excluded






