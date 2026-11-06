"""Offline checks for source coverage, bounded decoding, and intact 15-turn windows."""

import argparse
from contextlib import nullcontext
import copy
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wild.selection import build, helpers


def source(cid="one", turns=16, user="A source message"):
    return {
        "conversation_hash": cid, "language": "English", "model": "recorded-model",
        "conversation": [{"role": role, "content": f"{user} {index}", "turn_identifier": f"{cid}-{index // 2}"}
                         for index, role in enumerate(["user", "assistant"] * turns)],
    }


@pytest.mark.parametrize("word,domain", [
    ("depressed", "therapy"), ("depression", "therapy"), ("depressive", "therapy"),
    ("suicide", "therapy"), ("suicidal", "therapy"), ("heartbroken", "therapy"),
    ("heartbreak", "therapy"), ("sentient", "ai"), ("sentience", "ai"),
])
def test_retrieval_matches_complete_word_forms_without_assigning_a_domain(word, domain):
    row, reason = build.candidate(source(user=word), "pinned/file", 0, 3, 15)
    assert reason is None
    assert row["kw"] == [domain]
    assert "domain" not in row


def test_fifteen_turns_are_retained_from_source_without_editing_or_length_filtering():
    original = source(user="  literal text\nwith spacing  ")
    unchanged = copy.deepcopy(original)
    row, reason = build.candidate(original, "pinned/file", 17, 3, 15)
    assert reason is None
    assert original == unchanged
    assert row["source_turns"] == 16
    assert len(row["users"]) == len(row["gpt"]) == 15
    assert row["users"][12:] == [original["conversation"][index]["content"] for index in (24, 26, 28)]
    assert row["gpt"][14] == original["conversation"][29]["content"]
    assert len(build.candidate(source(turns=3), "pinned/file", 0, 3, 15)[0]["users"]) == 3


def test_seeded_subset_samples_across_listing_and_is_reproducible():
    files = [f"file-{index:03}.parquet" for index in range(86)]
    subset = build.source_subset(files, 12, 0)
    assert len(set(subset)) == 12
    assert subset == sorted(subset)
    assert subset == build.source_subset(list(reversed(files)), 12, 0)
    assert subset != build.source_subset(files, 12, 1)
    assert subset != files[:12]
    assert any(path >= "file-060.parquet" for path in subset)
    assert build.source_subset(files, 12, 0, "prefix") == files[:12]
    assert build.source_subset(files, None, 0) == files


def test_arrow_scan_decodes_small_batches_and_preserves_source_row_positions(monkeypatch):
    calls = []
    rows = [source("first"), source("excluded"), source("third")]
    rows[1]["language"] = "French"

    class ParquetFile:
        def __init__(self, stream):
            pass

        def iter_batches(self, **kwargs):
            calls.append(kwargs)
            for batch in (rows[:2], rows[2:]):
                yield types.SimpleNamespace(to_pylist=lambda batch=batch: batch)

    parquet = types.SimpleNamespace(ParquetFile=ParquetFile)
    monkeypatch.setitem(sys.modules, "pyarrow", types.SimpleNamespace(parquet=parquet))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    filesystem = types.SimpleNamespace(open=lambda path: nullcontext("stream"))
    result, dropped = build.scan(filesystem, "pinned/file", 3, 15)
    assert calls == [{"batch_size": 128, "use_threads": False, "columns": ["conversation_hash", "model", "language", "conversation"]}]
    assert [row["source_row"] for row in result] == [0, 2]
    assert dropped == {"language": 1}


def test_build_records_subset_coverage_and_rejects_reusing_a_changed_policy(tmp_path, monkeypatch):
    sha = "a" * 40
    files = [f"datasets/{build.DATASET}@{sha}/data/train-{index:03}.parquet" for index in range(86)]
    hub = types.SimpleNamespace(dataset_info=lambda *args, **kwargs: types.SimpleNamespace(sha=sha))
    filesystem = types.SimpleNamespace(glob=lambda pattern: list(reversed(files)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=lambda: hub, HfFileSystem=lambda: filesystem))
    scanned = []

    def scan(fs, path, min_turns, max_turns, batch_size):
        scanned.append((path, batch_size))
        row, _ = build.candidate(source(path), path, 0, min_turns, max_turns)
        return [row], {}

    monkeypatch.setattr(build, "scan", scan)
    args = argparse.Namespace(out=str(tmp_path / "candidates.jsonl"), dataset=build.DATASET, revision=sha, min_turns=3, max_turns=15, others=8000, random=None, max_files=12, seed=0, file_sampling="seeded", workers=2, scan_batch_size=128)
    result = build.run(args)
    assert result["source_files"] == build.source_subset(files, 12, 0)
    assert sorted(path for path, _ in scanned) == result["source_files"]
    assert {size for _, size in scanned} == {128}
    coverage = result["source_subset"]
    assert coverage["available_files"] == 86
    assert coverage["selected_files"] == 12
    assert coverage["file_fraction"] == 12 / 86
    assert coverage["complete_file_listing"] is False
    assert coverage["available_files_sha256"] == helpers.checksum(helpers.json_bytes(files))
    args.file_sampling = "prefix"
    with pytest.raises(ValueError, match="settings changed"):
        build.run(args)
    assert len(scanned) == 12


def test_cli_defaults_retain_fifteen_turns_and_use_bounded_scanning(monkeypatch):
    configs = []
    monkeypatch.setattr(build, "run", configs.append)
    build.main(["unused.jsonl"])
    assert configs[0].max_turns == 15
    assert configs[0].workers == 2
    assert configs[0].scan_batch_size == 128
    with pytest.raises(SystemExit):
        build.main(["unused.jsonl", "--workers", "8"])
