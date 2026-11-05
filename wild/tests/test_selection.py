"""Offline checks for the scientific selection contract and cache boundary."""

import argparse
import copy
import json
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build
import selection

SHA = "a" * 40


def conversation(cid="c", turns=3, prefix=None):
    return {
        "hash": cid,
        "users": [f"{prefix or cid} user {i}" for i in range(turns)],
        "gpt": [f"{prefix or cid} reply {i}" for i in range(turns)],
        "kw": [],
    }


def build_args(path, **changes):
    values = dict(out=str(path), dataset=build.DATASET, revision="main", min_turns=3, max_turns=12, others=8000, random=None, max_files=None, seed=0)
    return argparse.Namespace(**(values | changes))










def test_candidate_keeps_exact_initial_window():
    text = "  literal text\u2028and newline\n  "
    source = {"conversation_hash": "x", "language": "English", "model": "gpt", "conversation": [
        {"role": role, "content": text + str(i), "turn_identifier": i // 2}
        for i, role in enumerate(["user", "assistant"] * 5)
    ]}
    original = copy.deepcopy(source)
    result, reason = build.candidate(source, "source@sha/file", 12, 3, 4)
    assert reason is None
    assert source == original
    assert result["users"] == [source["conversation"][i]["content"] for i in (0, 2, 4, 6)]
    assert result["source_turns"] == 5
    assert result["source_turn_ids"] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert result["source_row"] == 12
    assert build.candidate(source, "file", 0, 6, 12)[1] == "too_short"


def test_dedup_keeps_longest_and_avoids_common_greeting_false_match():
    short = conversation("short", prefix="snapshot")
    long = conversation("long", turns=4, prefix="snapshot")
    other = conversation("other")
    other["users"][0], other["gpt"][0] = long["users"][0], long["gpt"][0]
    kept, excluded = selection.deduplicate([short, other, long, copy.deepcopy(long)])
    assert {row["hash"] for row in kept} == {"long", "other"}
    assert {row["reason"] for row in excluded} == {"duplicate_id", "duplicate_transcript_or_prefix"}
    assert selection.deduplicate(list(reversed([short, other, long, copy.deepcopy(long)]))) == (kept, excluded)


def test_shared_turn_ids_catch_branches_with_different_text():
    left, right = conversation("left"), conversation("right")
    left["source_turn_ids"], right["source_turn_ids"] = [123, None], [123, 456]
    kept, excluded = selection.deduplicate([right, left])
    assert len(kept) == 1
    assert excluded == [{"hash": "right", "reason": "shared_source_turn", "duplicate_of": "left"}]


















def test_build_resolves_revision_once_and_reuses_frozen_files(tmp_path, monkeypatch):
    paths = []
    class Hub:
        def dataset_info(self, repo, revision):
            return types.SimpleNamespace(sha=SHA)
    class Files:
        def glob(self, pattern):
            paths.append(pattern)
            return [pattern.replace("*", "train-0000")]
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Hub, HfFileSystem=Files))
    monkeypatch.setattr(build, "scan", lambda *_: ([conversation()], {}))
    args = build_args(tmp_path / "convs.jsonl")
    result = build.run(args)
    assert f"@{SHA}/" in paths[0]
    assert result["dataset"]["revision"] == SHA
    monkeypatch.setattr(build, "scan", lambda *_: pytest.fail("cache scanned dataset"))
    assert build.run(args) == result
    args.min_turns = 4
    with pytest.raises(ValueError, match="settings changed"):
        build.run(args)
    args.min_turns = 3
    monkeypatch.setattr(build, "BAD_STRINGS", (*build.BAD_STRINGS, "new control marker"))
    with pytest.raises(ValueError, match="settings changed"):
        build.run(args)


def test_locks_and_interrupted_publication(tmp_path):
    lock = tmp_path / "lock"
    with selection.selection_lock(lock):
        with pytest.raises(ValueError, match="another selection process"):
            with selection.selection_lock(lock):
                pass
    output = tmp_path / "selected.jsonl"
    selection.atomic_write(output, b"complete row\n")
    with pytest.raises(ValueError, match="without a completion manifest"):
        selection.load_completed(tmp_path / "manifest.json", {}, {"selected_sha256": output})
