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
sys.path.insert(0, str(WILD.parent))
from wild.drift import measurement as activations
from wild.drift import runner as drift_run
from wild.drift import generation as generate
from wild.drift import state as run_state


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
    monkeypatch.setattr(generate, "prompt_ids", lambda tok, messages: list(json.dumps(messages).encode()))
    monkeypatch.setattr(generate, "load_token_exclusions", lambda *args: [0])
    monkeypatch.setattr(generate, "drift_runtime", lambda: {"tensor_parallel_size": 1, "gpus": ["fake"]})
    monkeypatch.setattr(generate, "load_drift_model", lambda *args: object())
    monkeypatch.setattr(generate, "sample_batch", lambda llm, jobs, config: completions(jobs))
    monkeypatch.setattr(activations, "projection_runtime", lambda: {"device_map": "auto", "gpus": ["fake"]})
    monkeypatch.setattr(activations, "load_projector", lambda run: lambda messages: (1.25, 0.1))
    return tmp_path, records


def completions(jobs, text="new reply", token_ids=None):
    return [{"prompt_token_ids": job["prompt_token_ids"], "response_token_ids": [70, 71, 0] if token_ids is None else token_ids,
             "reply": text, "finish": "stop", "stop_reason": 0} for job in jobs]


def run_both(path):
    generate.main([str(path)])
    activations.main([str(path)])


def forbid(*args, **kwargs):
    raise AssertionError("cached run attempted a model load or network operation")


def test_cli_runs_and_reuses_cache_without_loading(experiment, monkeypatch):
    path, records = experiment
    run_both(path)
    assert len(run_state.jsonl((path / "drift.jsonl").read_bytes())) == 18
    assert len(run_state.jsonl((path / "projections.jsonl").read_bytes())) == 18
    monkeypatch.setattr(run_state, "resolve_artifacts", forbid)
    monkeypatch.setattr(generate, "load_drift_tokenizer", forbid)
    monkeypatch.setattr(generate, "load_drift_model", forbid)
    monkeypatch.setattr(activations, "load_projector", forbid)
    run_both(path)


def test_drift_only_cli_does_not_require_harmbench(experiment):
    from wild import generate as legacy_generate

    path, _ = experiment
    legacy_generate.main([str(path), "--drift-only"])
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
        monkeypatch.setattr(run_state, "settings", lambda horizon, samples_per_turn=3: {**original(horizon, samples_per_turn), "temperature": 0.1})
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
    monkeypatch.setattr(generate, "sample_batch", fail)
    with pytest.raises(RuntimeError):
        generate.run_drift(path)
    assert not (path / "drift.jsonl").exists()
    assert run_state.read_record(path / "drift.state.json")["status"] == "pending"
    monkeypatch.setattr(generate, "sample_batch", lambda llm, jobs, config: completions(jobs, "retry"))
    generate.run_drift(path)
    assert len(run_state.jsonl((path / "drift.jsonl").read_bytes())) == 18


def test_complete_responses_without_completion_record_are_reused(experiment, monkeypatch):
    path, _ = experiment
    generate.run_drift(path)
    state = run_state.read_record(path / "drift.state.json")
    state["status"] = "pending"
    run_state.save_record(path / "drift.state.json", state)
    monkeypatch.setattr(generate, "load_drift_model", forbid)
    generate.run_drift(path)
    assert run_state.read_record(path / "drift.state.json")["status"] == "complete"


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
    assert state["plan"]["generated_keys"] == [[cid, 1, i] for cid in ("a", "b") for i in range(3)]
    assert [row["reason"] for row in state["plan"]["excluded"]] == [
        reason for _ in range(2) for reason in (["prompt_too_long"] * 3 + ["after_overlength_prefix"] * 3)
    ]
    generate.run_drift(path)


def test_projection_resumes_successes_and_keeps_original_history(experiment, monkeypatch):
    path, _ = experiment
    generate.run_drift(path)
    calls = []
    def project(messages):
        calls.append(messages)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return 2.0, 0.0
    monkeypatch.setattr(activations, "load_projector", lambda run: project)
    with pytest.raises(RuntimeError):
        activations.run_projection(path)
    assert len(run_state.jsonl((path / "projections.jsonl").read_bytes())) == 1
    calls.clear()
    monkeypatch.setattr(activations, "load_projector", lambda run: lambda messages: (calls.append(messages) or (2.0, 0.0)))
    activations.run_projection(path)
    assert len(calls) == 17
    turn_two = next(row for row in calls if row["hash"] == "a" and row["k"] == 2)
    messages = json.loads(bytes(turn_two["prompt_token_ids"]))
    assert messages[1]["content"] == "a original 0"
    assert messages[-1]["content"] == "a user 1"
    assert turn_two["reply"] == "new reply"


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "mixed_run", "nonfinite", "invalid_fire", "null", "checksum"])
def test_invalid_projection_checkpoints_rejected(experiment, monkeypatch, mutation):
    path, _ = experiment
    run_both(path)
    p = path / "projections.jsonl"
    rows = run_state.jsonl(p.read_bytes())
    state = run_state.read_record(path / "projections.state.json")
    state["status"] = "pending"
    run_state.save_record(path / "projections.state.json", state)
    if mutation == "duplicate":
        rows.append(rows[0])
    else:
        row = rows[0]
        if mutation == "unknown": row["hash"] = "unknown"
        elif mutation == "mixed_run": row["run_id"] = "other"
        elif mutation == "nonfinite": row["proj"] = "NaN"
        elif mutation == "invalid_fire": row["fire_frac"] = 1.1
        elif mutation == "null": row.update(proj=None, fire_frac=None, reason="no_content_tokens")
        elif mutation == "checksum": row["proj"] = 10
        if mutation != "checksum":
            row["record_sha256"] = run_state.fingerprint({k: v for k, v in row.items() if k != "record_sha256"})
    p.write_bytes(b"".join(run_state.encoded(row) for row in rows))
    monkeypatch.setattr(activations, "load_projector", forbid)
    with pytest.raises(ValueError):
        activations.run_projection(path)


def test_interrupted_projection_line_fails_without_discarding_it(experiment):
    path, _ = experiment
    run_both(path)
    state = run_state.read_record(path / "projections.state.json")
    state["status"] = "pending"
    run_state.save_record(path / "projections.state.json", state)
    p = path / "projections.jsonl"
    damaged = p.read_bytes() + b'{"hash":'
    p.write_bytes(damaged)
    with pytest.raises(ValueError, match="interrupted final write"):
        activations.run_projection(path)
    assert p.read_bytes() == damaged


def test_empty_response_has_explicit_null_reason(experiment, monkeypatch):
    path, _ = experiment
    monkeypatch.setattr(generate, "sample_batch", lambda llm, jobs, config: completions(jobs, "", [0]))
    monkeypatch.setattr(activations, "load_projector", forbid)
    run_both(path)
    rows = run_state.jsonl((path / "projections.jsonl").read_bytes())
    assert all(row["proj"] is None and row["reason"] == "no_content_tokens" for row in rows)
    run_both(path)


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


def test_extraction_passes_same_pinned_revision_to_hf_loaders(monkeypatch):
    calls = []
    class Vector:
        def float(self): return self
        def norm(self): return 1
    vector = Vector()
    functional = types.SimpleNamespace(normalize=lambda value, dim: value)
    fake_torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(functional=functional), bfloat16="bf16",
        isfinite=lambda value: types.SimpleNamespace(all=lambda: True),
        backends=types.SimpleNamespace(cuda=types.SimpleNamespace(enable_cudnn_sdp=lambda enabled: None)),
    )
    tokenizer = types.SimpleNamespace(pad_token=None, eos_token="end")
    model = types.SimpleNamespace(eval=lambda: None)
    def tokenizer_load(*args, **kwargs):
        calls.append(("tokenizer", args, kwargs))
        return tokenizer
    def model_load(*args, **kwargs):
        calls.append(("model", args, kwargs))
        return model
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_torch.nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functional)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizer_load),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=model_load),
    ))
    cfg = {"vectors": {"cap": {"layer": 56, "vector": vector}}, "experiments": [
        {"id": "cap", "interventions": [{"vector": "cap", "cap": 1.0}]}
    ]}
    monkeypatch.setitem(sys.modules, "assistant_axis", types.SimpleNamespace(
        load_axis=lambda path: [vector] * 41, load_capping_config=lambda path: cfg,
    ))
    monkeypatch.setitem(sys.modules, "assistant_axis.internals", types.SimpleNamespace(
        ConversationEncoder=lambda *args: object(), StopForward=RuntimeError,
        ProbingModel=types.SimpleNamespace(from_existing=lambda *args, **kwargs: types.SimpleNamespace(get_layers=lambda: [])),
    ))
    paths = []
    monkeypatch.setattr(run_state, "artifact_path", lambda value: paths.append(value) or value["filename"])
    artifacts = {
        "model": {"repo_id": "test/llama", "revision": "b" * 40},
        "axis": {"filename": "axis.pt", "revision": "c" * 40},
        "capping": {"filename": "cap.pt", "revision": "c" * 40},
    }
    activations.load_projector({"artifacts": artifacts, "config": {"layer": 40, "capping_experiment": "cap"}})
    assert [call[0] for call in calls] == ["tokenizer", "model"]
    assert all(call[2]["revision"] == "b" * 40 for call in calls)
    assert paths == [artifacts["axis"], artifacts["capping"]]


def test_cli_help_has_no_gpu_import_requirement(tmp_path):
    for filename in ("generate.py", "drift_activations.py", "drift_run.py", "drift_analysis.py"):
        result = subprocess.run([sys.executable, str(WILD / filename), "--help"], cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_runner_uses_separate_checked_processes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(drift_run.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    drift_run.main([str(tmp_path), "--model-revision", "b" * 40, "--axis-revision", "c" * 40])
    assert len(calls) == 3
    assert calls[0][0][0] == sys.executable
    assert Path(calls[0][0][1]).name == "generate.py"
    assert "--drift-only" in calls[0][0]
    assert "b" * 40 in calls[0][0] and "c" * 40 in calls[0][0]
    assert Path(calls[1][0][1]).name == "drift_activations.py"
    assert Path(calls[2][0][1]).name == "drift_analysis.py"
    assert all(kwargs == {"check": True} for _, kwargs in calls)


def test_runner_stops_after_failed_generation(monkeypatch, tmp_path):
    calls = []
    def fail(args, **kwargs):
        calls.append(args)
        raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(drift_run.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        drift_run.main([str(tmp_path)])
    assert len(calls) == 1


@pytest.mark.parametrize("protected", [False, True])
def test_legacy_runner_dispatches_correct_projector(tmp_path, protected):
    if protected:
        (tmp_path / "drift.manifest.json").write_text("{}")
    # Execute the exact dispatch block with echo as the model command; the rest
    # of run.sh (installation, downloads, pod lifecycle) must never run in tests.
    script = (WILD / "run.sh").read_text()
    dispatch = script.split("step activations\n", 1)[1].split("# step capability;", 1)[0]
    result = subprocess.run(
        ["bash", "-c", dispatch], capture_output=True, text=True, check=True,
        env={**os.environ, "RUN": str(tmp_path), "PY": "echo"},
    )
    expected = "drift_activations.py" if protected else "activations.py"
    assert result.stdout.strip() == f"{expected} {tmp_path}"


@pytest.mark.parametrize("protected,has_output,should_generate", [(True, True, True), (True, False, True), (False, False, True), (False, True, False)])
def test_combined_runner_reenters_protected_generation(tmp_path, protected, has_output, should_generate):
    if protected:
        (tmp_path / "drift.manifest.json").write_text("{}")
    if has_output:
        (tmp_path / "drift.jsonl").write_text("partial or completed output")
    script = (WILD / "run.sh").read_text()
    block = script.split('# Protected outputs may be partial checkpoints;', 1)[1].split('# Harm on real multi-turn', 1)[0]
    block = block[block.index('if ['):]
    result = subprocess.run(
        ["bash", "-c", 'step() { :; }\n' + block], capture_output=True, text=True, check=True,
        env={**os.environ, "RUN": str(tmp_path), "PY": "echo", "DL": "/unused"},
    )
    assert ("generate.py" in result.stdout) is should_generate
