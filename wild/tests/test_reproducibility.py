"""Offline integration checks for frozen drift runs; no model downloads."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

WILD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WILD))
import generate
import run_state


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    records = [
        {"hash": cid, "domain": domain,
         "users": [f"{cid} user {k}" for k in range(3)],
         "gpt": [f"{cid} original {k}" for k in range(3)]}
        for cid, domain in (("a", "coding"), ("b", "therapy"))
    ]
    raw = b"".join(run_state.encoded(r) for r in records)
    (tmp_path / "selected.jsonl").write_bytes(raw)
    sidecar = {
        "schema_version": 1, "selected_sha256": run_state.checksum(raw),
        "dataset": {"repo_id": "source/WildChat", "revision": "a" * 40},
        "selection": {"max_turns": 3, "min_turns": 3}, "review_status": "unreviewed",
    }
    sidecar["manifest_sha256"] = run_state.fingerprint(sidecar)
    (tmp_path / "selected.manifest.json").write_bytes(run_state.encoded(sidecar))
    artifacts = {
        "model": {"repo_id": "test/llama", "revision": "b" * 40},
        "axis": {"repo_id": "test/vectors", "revision": "c" * 40, "filename": "axis.pt", "sha256": "axis"},
        "capping": {"repo_id": "test/vectors", "revision": "c" * 40, "filename": "cap.pt", "sha256": "cap"},
    }
    monkeypatch.setattr(run_state, "resolve_artifacts", lambda *args: copy.deepcopy(artifacts))
    monkeypatch.setattr(run_state, "environment", lambda stage: {"test-runtime": "1"})
    monkeypatch.setattr(generate, "load_drift_tokenizer", lambda artifact: object())
    monkeypatch.setattr(generate, "prompt_ids", lambda tok, messages: [1] * len(messages))
    monkeypatch.setattr(generate, "drift_runtime", lambda: {"tensor_parallel_size": 1, "gpus": ["fake"]})
    monkeypatch.setattr(generate, "load_drift_model", lambda *args: object())
    monkeypatch.setattr(generate, "generate", lambda llm, prompts, max_tokens: [[("new reply", "stop")] for _ in prompts])
    return tmp_path, records




def forbid(*args, **kwargs):
    raise AssertionError("cached run attempted a model load or network operation")




def test_drift_only_cli_does_not_require_harmbench(experiment):
    path, _ = experiment
    generate.main([str(path), "--drift-only"])
    assert (path / "drift.jsonl").exists()


@pytest.mark.parametrize("change", ["input", "sidecar", "settings", "code", "packages", "requested_revision"])
def test_changed_experiment_rejects_cache(experiment, monkeypatch, change):
    path, _ = experiment
    generate.run_drift(path)
    options = {}
    if change == "input":
        (path / "selected.jsonl").write_bytes((path / "selected.jsonl").read_bytes().replace(b"a user", b"changed user"))
    elif change == "sidecar":
        p = path / "selected.manifest.json"
        value = json.loads(p.read_bytes())
        value["review_status"] = "reviewed"
        p.write_bytes(run_state.encoded(value))
    elif change == "settings":
        original = run_state.settings
        monkeypatch.setattr(run_state, "settings", lambda horizon: {**original(horizon), "temperature": 0.1})
    elif change == "code":
        monkeypatch.setattr(run_state, "code_fingerprints", lambda stage: {"changed": "code"})
    elif change == "packages":
        monkeypatch.setattr(run_state, "environment", lambda stage: {"test-runtime": "2"})
    else:
        options["model_revision"] = "d" * 40
    monkeypatch.setattr(generate, "load_drift_tokenizer", forbid)
    with pytest.raises(ValueError):
        generate.run_drift(path, **options)


def test_manifest_tampering_is_detected(experiment):
    path, _ = experiment
    generate.run_drift(path)
    p = path / "drift.manifest.json"
    value = json.loads(p.read_bytes())
    value["artifacts"]["axis"]["revision"] = "d" * 40
    p.write_bytes(run_state.encoded(value))
    with pytest.raises(ValueError, match="fingerprint"):
        generate.run_drift(path)


def test_legacy_output_is_not_adopted(experiment, monkeypatch):
    path, _ = experiment
    (path / "drift.jsonl").write_text("{}\n")
    monkeypatch.setattr(run_state, "resolve_artifacts", forbid)
    with pytest.raises(ValueError, match="unmanifested"):
        generate.run_drift(path)


def test_interrupted_generation_never_publishes_partial_results(experiment, monkeypatch):
    path, _ = experiment
    def fail(*args):
        raise RuntimeError("generation interrupted")
    monkeypatch.setattr(generate, "generate", fail)
    with pytest.raises(RuntimeError):
        generate.run_drift(path)
    assert not (path / "drift.jsonl").exists()
    assert run_state.read_record(path / "drift.state.json")["status"] == "pending"
    monkeypatch.setattr(generate, "generate", lambda llm, prompts, n: [[("retry", "stop")] for _ in prompts])
    generate.run_drift(path)
    assert len(run_state.jsonl((path / "drift.jsonl").read_bytes())) == 6


def test_file_without_completion_is_not_treated_as_done(experiment):
    path, _ = experiment
    generate.run_drift(path)
    state = run_state.read_record(path / "drift.state.json")
    state["status"] = "pending"
    run_state.save_record(path / "drift.state.json", state)
    with pytest.raises(ValueError, match="publication was interrupted"):
        generate.run_drift(path)


def test_output_corruption_is_detected(experiment):
    path, _ = experiment
    generate.run_drift(path)
    with (path / "drift.jsonl").open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(ValueError, match="checksum"):
        generate.run_drift(path)


def test_turn_exclusions_are_explicit_and_horizon_comes_from_selection(experiment, monkeypatch):
    path, _ = experiment
    monkeypatch.setattr(generate, "prompt_ids", lambda tok, messages: [0] * (20000 if len(messages) >= 3 else 1))
    generate.run_drift(path)
    state = run_state.read_record(path / "drift.state.json")
    assert state["plan"]["generated_keys"] == [["a", 1], ["b", 1]]
    assert [row["reason"] for row in state["plan"]["excluded"]] == [
        "prompt_too_long", "after_overlength_prefix", "prompt_too_long", "after_overlength_prefix"
    ]
    generate.run_drift(path)










def test_output_lock_rejects_concurrent_writers(tmp_path):
    with run_state.run_lock(tmp_path):
        with pytest.raises(ValueError, match="another drift process"):
            with run_state.run_lock(tmp_path):
                pass


def test_generation_passes_pinned_revisions_to_loaders(monkeypatch):
    calls = []
    fake_transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: calls.append(("tokenizer", args, kwargs))
    ))
    fake_vllm = types.SimpleNamespace(LLM=lambda *args, **kwargs: calls.append(("model", args, kwargs)))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    artifact = {"repo_id": "test/llama", "revision": "b" * 40}
    generate.load_drift_tokenizer(artifact)
    generate.load_drift_model(artifact, run_state.settings(3), {"tensor_parallel_size": 2})
    assert calls[0][2]["revision"] == "b" * 40
    assert calls[1][2]["revision"] == calls[1][2]["tokenizer_revision"] == "b" * 40


def test_vector_download_uses_pinned_revision_and_checks_bytes(tmp_path, monkeypatch):
    path = tmp_path / "axis.pt"
    path.write_bytes(b"axis bytes")
    calls = []
    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download))
    artifact = {"repo_id": "test/vectors", "filename": "axis.pt", "revision": "c" * 40,
                "sha256": run_state.checksum(path.read_bytes())}
    assert run_state.artifact_path(artifact) == str(path)
    assert calls[0][1]["revision"] == "c" * 40
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        run_state.artifact_path(artifact)










