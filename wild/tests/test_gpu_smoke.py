"""Offline contracts for the real-GPU smoke driver; no model downloads."""

import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wild.drift import smoke, state
from wild.selection import helpers


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        text = json.dumps(messages)
        return list(text.encode()) if kwargs.get("tokenize", True) else text

    def __call__(self, text, **kwargs):
        return types.SimpleNamespace(input_ids=list(text.encode()))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    rows = [{"hash": name, "users": [f"{name} user {k}" for k in range(3)],
             "gpt": [f"{name} ORIGINAL assistant {k}" for k in range(3)]}
            for name in ("a", "b", "c", "d")]
    pool = tmp_path / "candidates.jsonl"
    raw = helpers.jsonl_bytes(rows)
    pool.write_bytes(raw)
    manifest = helpers.seal_manifest({
        "schema_version": 1, "dataset": {"repo_id": "fixture/data", "revision": "d" * 40},
        "config": {"min_turns": 3, "max_turns": 3}, "candidates_sha256": state.checksum(raw),
    })
    Path(str(pool) + ".manifest.json").write_bytes(helpers.json_bytes(manifest))
    artifacts = {
        "model": {"repo_id": "fixture/model", "revision": "a" * 40},
        **{name: {"repo_id": "fixture/vectors", "revision": "b" * 40,
                  "filename": name + ".pt", "sha256": "c" * 64}
           for name in ("axis", "capping")},
    }
    monkeypatch.setattr(state, "resolve_artifacts", lambda *args: artifacts)
    monkeypatch.setattr(smoke.generation, "load_drift_tokenizer", lambda *args: Tokenizer())
    monkeypatch.setattr(smoke.generation, "load_token_exclusions", lambda *args: [0])
    monkeypatch.setattr(smoke.generation, "drift_runtime", lambda: {"tensor_parallel_size": 2, "gpus": ["fake", "fake"]})
    monkeypatch.setattr(smoke.measurement, "projection_runtime", lambda: {"gpus": ["fake", "fake"]})
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", types.SimpleNamespace(StructuredOutputsParams=lambda **kwargs: kwargs))

    def complete(prompts, params):
        if isinstance(params, dict):
            assert params == {"temperature": 0, "max_tokens": 256, "seed": 0, "structured_outputs": {"json": helpers.LABEL_SCHEMA, "disable_any_whitespace": True}}
            for prompt in prompts:
                messages = json.loads(bytes(prompt["prompt_token_ids"]))
                assert messages[0]["content"] == helpers.POLICY
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(
                text='{"domain":"other","subtopic":null,"rationale":"Diagnostic fixture."}', finish_reason="stop", stop_reason=None, token_ids=[1, 2, 3])]) for _ in prompts]
        assert len(prompts) == len(params) == 6
        assert all(p["max_tokens"] == 1024 for p in params)
        return [types.SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], outputs=[types.SimpleNamespace(
            token_ids=[301, 302, 0], text="Deliberately unrelated to the token IDs.",
            finish_reason="stop", stop_reason=0)]) for prompt in prompts]

    monkeypatch.setattr(smoke.generation, "load_drift_model", lambda *args: types.SimpleNamespace(generate=complete))
    projected = []
    def projector(row):
        projected.append(row)
        return 1.25, 0.5
    monkeypatch.setattr(smoke.measurement, "load_projector", lambda run: projector)
    calls = []
    def worker(command, **kwargs):
        assert command[:5] == [sys.executable, "-u", "-m", "wild.drift.smoke", "_worker"]
        assert kwargs == {"check": True, "cwd": state.ROOT}
        calls.append(command[5])
        smoke.main(command[4:])
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(smoke.subprocess, "run", worker)
    return pool, tmp_path / "smoke", rows, projected, calls


def test_end_to_end_smoke_keeps_diagnostics_separate_and_exact_tokens(fixture):
    pool, run_dir, _, projected, calls = fixture
    result = smoke.run_smoke(pool, run_dir, "a" * 40)
    assert calls == ["generation", "projection"]
    assert result["status"] == "complete"
    assert set(result["stages"].values()) == {"complete"}
    assert result["artifacts"]["axis"]["revision"] == "b" * 40
    assert result["config"] == state.settings(3)
    assert state.read_record(run_dir / "smoke.result.json") == result
    assert all(path.name.startswith("smoke.") for path in run_dir.iterdir())
    assert result["outputs"] == smoke.output_checksums(run_dir, result["outputs"])
    replies = state.jsonl((run_dir / "smoke.replies.jsonl").read_bytes())
    assert len(replies) == 6
    assert all(row["response_token_ids"] == [301, 302, 0] for row in replies)
    assert all("domain" not in row and row["response_token_count"] == 2 for row in replies)
    assert projected[:6] == replies
    stress = state.read_record(run_dir / "smoke.stress.json")
    assert stress["synthetic"] is True and stress["input_tokens"] == 16384
    assert stress["last_measured_position"] == 16383
    assert len(projected[-1]["prompt_token_ids"]) == 15360
    assert len(projected[-1]["response_token_ids"]) == 1024
    assert set(projected[-1]["response_token_ids"]) == {301, 302}


def test_plan_is_order_independent_original_history_and_three_seeded_samples(fixture):
    _, _, rows, _, _ = fixture
    plan = smoke.plan(rows, Tokenizer(), state.settings(3))
    assert plan == smoke.plan(list(reversed(rows)), Tokenizer(), state.settings(3))
    assert len(plan["prefixes"][0]["prompt_token_ids"]) < len(plan["prefixes"][1]["prompt_token_ids"])
    by_hash = {row["hash"]: row for row in rows}
    for prefix in plan["prefixes"]:
        row = by_hash[prefix["hash"]]
        assert json.loads(bytes(prefix["prompt_token_ids"])) == smoke.generation.history(row["users"], row["gpt"], prefix["k"])
        jobs = [job for job in plan["jobs"] if (job["hash"], job["k"]) == (prefix["hash"], prefix["k"])]
        assert {job["sample_index"] for job in jobs} == {0, 1, 2}
        assert len({job["seed"] for job in jobs}) == 3


@pytest.mark.parametrize("failure", ["checksum", "manifest", "duplicates"])
def test_bad_candidate_provenance_fails_before_model_or_subprocess(fixture, failure):
    pool, run_dir, rows, _, calls = fixture
    sidecar = Path(str(pool) + ".manifest.json")
    manifest = json.loads(sidecar.read_bytes())
    if failure == "checksum":
        pool.write_bytes(pool.read_bytes() + b"\n")
    elif failure == "manifest":
        manifest["config"]["max_turns"] += 1
        sidecar.write_bytes(helpers.json_bytes(manifest))
    else:
        raw = helpers.jsonl_bytes(rows + [rows[0]])
        pool.write_bytes(raw)
        manifest["candidates_sha256"] = state.checksum(raw)
        sidecar.write_bytes(helpers.json_bytes(helpers.seal_manifest(manifest)))
    with pytest.raises(ValueError):
        smoke.run_smoke(pool, run_dir, "a" * 40)
    assert calls == []
    result = state.read_record(run_dir / "smoke.result.json")
    assert result["status"] == "failed" and result["stages"]["prepare"] == "failed"


def test_existing_directory_is_never_adopted_or_overwritten(fixture):
    pool, run_dir, _, _, _ = fixture
    run_dir.mkdir()
    sentinel = run_dir / "smoke.result.json"
    sentinel.write_text("preserve me")
    with pytest.raises(ValueError, match="new directory"):
        smoke.run_smoke(pool, run_dir, "a" * 40)
    assert sentinel.read_text() == "preserve me"


def test_unparsed_judge_output_is_saved_but_cannot_pass(fixture, monkeypatch):
    pool, run_dir, _, _, calls = fixture
    fake = types.SimpleNamespace(generate=lambda prompts, params: [types.SimpleNamespace(
        outputs=[types.SimpleNamespace(text="not valid JSON", finish_reason="stop", stop_reason=None, token_ids=[1, 2, 3])]) for _ in prompts])
    monkeypatch.setattr(smoke.generation, "load_drift_model", lambda *args: fake)
    with pytest.raises(ValueError, match="strict judge"):
        smoke.run_smoke(pool, run_dir, "a" * 40)
    assert calls == ["generation"]
    assert all(row["status"] == "unparsed" for row in state.jsonl((run_dir / "smoke.labels.jsonl").read_bytes()))
    assert state.read_record(run_dir / "smoke.result.json")["status"] == "failed"


@pytest.mark.parametrize("failure", ["real_projection", "stress", "bad_fraction"])
def test_projection_or_full_context_failure_cannot_mark_smoke_complete(fixture, monkeypatch, failure):
    pool, run_dir, _, _, _ = fixture
    def project(row):
        if failure == "real_projection" or (failure == "stress" and row.get("synthetic")):
            return float("nan"), 0.5
        return (1.0, 1.1) if failure == "bad_fraction" else (1.0, 0.0)
    monkeypatch.setattr(smoke.measurement, "load_projector", lambda run: project)
    with pytest.raises(ValueError, match="nonfinite projection or invalid cap fraction"):
        smoke.run_smoke(pool, run_dir, "a" * 40)
    result = state.read_record(run_dir / "smoke.result.json")
    assert result["status"] == "failed"
    assert result["stages"]["generation"] == "complete" and result["stages"]["projection"] == "failed"
    assert not (run_dir / "smoke.projection.json").exists()


def test_generation_exit_failure_never_starts_hf(fixture, monkeypatch):
    pool, run_dir, _, _, calls = fixture
    def fail(command, **kwargs):
        calls.append(command[5])
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(smoke.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        smoke.run_smoke(pool, run_dir, "a" * 40)
    assert calls == ["generation"]
    result = state.read_record(run_dir / "smoke.result.json")
    assert result["status"] == "failed" and result["stages"]["projection"] == "pending"
