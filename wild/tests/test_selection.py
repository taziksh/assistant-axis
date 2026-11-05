"""Offline checks for the scientific selection contract and cache boundary."""

import argparse
import copy
import json
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wild.selection import build as build
from wild.selection import judge as judge
from wild.selection import helpers as selection

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


def judge_args(path, output, **changes):
    values = dict(convs=str(path), run_dir=str(output), per_domain=1, min_turns=3, max_turns=12, seed=0, model_revision="main", context_limit=16384, output_tokens=256)
    return argparse.Namespace(**(values | changes))


def source_pool(path, rows):
    raw = selection.jsonl_bytes(rows)
    selection.atomic_write(path, raw)
    manifest = {"schema_version": 1, "dataset": {"repo_id": build.DATASET, "revision": SHA}, "config": {"min_turns": 3, "max_turns": 12}, "candidates_sha256": selection.checksum(raw)}
    selection.atomic_write(str(path) + ".manifest.json", selection.json_bytes(selection.seal_manifest(manifest)))


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        # Real prompt data, not mocked length independent of the contents.
        return list(json.dumps(messages, ensure_ascii=False).encode())


def model_loader(*args):
    return Tokenizer(), lambda prompts: [json.dumps({"domain": "coding", "rationale": "Programming request."}) for prompt in prompts]


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


def test_prompt_contains_full_user_messages_and_no_assistant_text():
    row = conversation()
    row["users"][0] = "a" * 600 + " Important late topic change."
    prompt = selection.judge_prompt(row["users"])
    assert json.loads(prompt[1]["content"])["user_messages"] == row["users"]
    assert row["gpt"][0] not in prompt[1]["content"]


@pytest.mark.parametrize("raw", ["coding", '{"domain":"coding"}', '{"domain":[],"rationale":"a"}', '{"domain":"coding","rationale":""}', '{"domain":"coding","rationale":"a","extra":1}'])
def test_invalid_judge_outputs_are_excluded(raw):
    assert selection.parse_label(raw)["status"] == "unparsed"


def test_full_overlength_window_is_excluded_without_calling_judge():
    labels = judge.classify([conversation()], Tokenizer(), lambda _: pytest.fail("must not generate"), context_limit=100, output_tokens=10)
    assert labels[0]["status"] == "too_long"
    assert labels[0]["domain"] is None


def test_selection_is_deterministic_and_reports_exclusions_and_shortfall():
    rows = [conversation(str(i)) for i in range(5)]
    domains = ["coding", "coding", "roleplay", "mixed", "therapy"]
    labels = [{"hash": row["hash"], **selection.parse_label(json.dumps({"domain": domain, "rationale": domain}))} for row, domain in zip(rows, domains)]
    chosen, audit, counts = selection.select_domains(rows, labels, 1, 9)
    assert selection.select_domains(list(reversed(rows)), list(reversed(labels)), 1, 9) == (chosen, audit, counts)
    assert counts["selected_counts"] == {"coding": 1, "writing": 0, "therapy": 1, "ai": 0}
    assert counts["shortfalls"] == {"coding": 0, "writing": 1, "therapy": 0, "ai": 1}
    assert {row["outcome"] for row in audit} == {"selected", "quota_filled", "roleplay", "mixed"}


def test_judge_end_to_end_and_cache_are_offline(tmp_path):
    path, output = tmp_path / "convs.jsonl", tmp_path / "run"
    rows = [conversation("one"), conversation("two")]
    source_pool(path, rows)
    args = judge_args(path, output)
    result = judge.run(args, model_loader=model_loader, revision_resolver=lambda *_: SHA)
    assert result["dataset"]["revision"] == SHA
    assert result["judge"]["revision"] == SHA
    assert result["review_status"] == "unreviewed"
    assert result["counts"]["shortfalls"]["therapy"] == 1
    saved = selection.read_jsonl(output / "selected.jsonl")[0]
    assert saved["users"] in [row["users"] for row in rows]
    assert judge.run(args, model_loader=lambda *_: pytest.fail("cached selection loaded model"), revision_resolver=lambda *_: pytest.fail("cache resolved floating main")) == result


@pytest.mark.parametrize("change", ["input", "policy", "revision", "configuration", "runtime", "modified_output", "partial_output"])
def test_cached_selection_rejects_changes(tmp_path, monkeypatch, change):
    path, output = tmp_path / "convs.jsonl", tmp_path / "run"
    source_pool(path, [conversation()])
    args = judge_args(path, output)
    judge.run(args, model_loader=model_loader, revision_resolver=lambda *_: SHA)
    if change == "input":
        source_pool(path, [conversation("new")])
    elif change == "policy":
        monkeypatch.setattr(selection, "POLICY", selection.POLICY + "new")
    elif change == "revision":
        args.model_revision = "b" * 40
    elif change == "configuration":
        args.per_domain = 2
    elif change == "runtime":
        monkeypatch.setattr(judge, "runtime_versions", lambda: {"changed": "version"})
    elif change == "modified_output":
        (output / "selected.jsonl").write_text("[]\n")
    else:
        (output / "labels.jsonl").unlink()
    with pytest.raises(ValueError):
        judge.run(args, model_loader=lambda *_: pytest.fail("loaded model before cache check"), revision_resolver=lambda *_: pytest.fail("resolved before cache check"))


def test_legacy_output_cannot_be_adopted(tmp_path):
    path, output = tmp_path / "convs.jsonl", tmp_path / "run"
    source_pool(path, [conversation()])
    output.mkdir()
    (output / "labels.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="without a completion manifest"):
        judge.run(judge_args(path, output), model_loader=model_loader, revision_resolver=lambda *_: SHA)


@pytest.mark.parametrize("field", ["judge", "dataset", "counts", "review_status", "source_dataset"])
def test_manifest_provenance_tampering_is_rejected(tmp_path, field):
    path, output = tmp_path / "convs.jsonl", tmp_path / "run"
    source_pool(path, [conversation()])
    args = judge_args(path, output)
    judge.run(args, model_loader=model_loader, revision_resolver=lambda *_: SHA)
    manifest_path = Path(str(path) + ".manifest.json") if field == "source_dataset" else output / "selected.manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if field in {"judge", "dataset", "source_dataset"}:
        manifest["dataset" if field == "source_dataset" else field]["revision"] = "b" * 40
    elif field == "counts":
        manifest["counts"]["selected_counts"]["coding"] = 100
    else:
        manifest["review_status"] = "approved"
    selection.atomic_write(manifest_path, selection.json_bytes(manifest))
    with pytest.raises(ValueError, match="fingerprint"):
        judge.run(args, model_loader=lambda *_: pytest.fail("loaded model before metadata check"), revision_resolver=lambda *_: pytest.fail("resolved before metadata check"))


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
