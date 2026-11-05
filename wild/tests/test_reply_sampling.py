"""Sample identity, interruption recovery, and exact generated-token measurement."""

import json
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wild.drift import measurement as drift_activations
from wild.drift import generation as generate
from wild.drift import state as run_state
from test_reproducibility import completions, experiment, forbid, run_both


def test_three_samples_have_stable_independent_seeds(experiment):
    path, records = experiment
    config = run_state.settings(3)
    jobs, plan = generate.drift_plan(records, object(), config)
    reverse_jobs, reverse_plan = generate.drift_plan(list(reversed(records)), object(), config)
    assert jobs == reverse_jobs and plan == reverse_plan
    assert len(jobs) == 18
    for cid in ("a", "b"):
        for turn in (1, 2, 3):
            samples = [job for job in jobs if (job["hash"], job["k"]) == (cid, turn)]
            assert {job["sample_index"] for job in samples} == {0, 1, 2}
            assert len({job["seed"] for job in samples}) == 3
            assert len({tuple(job["prompt_token_ids"]) for job in samples}) == 1


def test_resume_only_generates_samples_missing_after_interruption(experiment, monkeypatch):
    path, _ = experiment
    batches = []
    def interrupted(llm, jobs, config):
        batches.append([run_state.row_key(job) for job in jobs])
        if len(batches) == 2:
            raise RuntimeError("interrupted second batch")
        return completions(jobs)
    monkeypatch.setattr(generate, "sample_batch", interrupted)
    with pytest.raises(RuntimeError, match="second batch"):
        generate.run_drift(path)
    saved = run_state.jsonl((path / "drift.jsonl").read_bytes())
    assert len(saved) == 16
    calls = []
    monkeypatch.setattr(generate, "sample_batch", lambda llm, jobs, config: calls.extend(jobs) or completions(jobs))
    generate.run_drift(path)
    assert len(calls) == 2
    assert not ({run_state.row_key(row) for row in saved} & {run_state.row_key(job) for job in calls})
    complete = run_state.jsonl((path / "drift.jsonl").read_bytes())
    assert complete[:16] == saved
    assert len(complete) == 18


def test_sample_count_is_recorded_reused_and_cannot_change(experiment):
    path, _ = experiment
    generate.run_drift(path, samples_per_turn=1)
    assert run_state.read_record(path / "drift.manifest.json")["config"]["samples_per_turn"] == 1
    generate.run_drift(path)
    drift_activations.run_projection(path)
    assert len(run_state.load_completed_run(path)[3]) == 6
    with pytest.raises(ValueError, match="settings changed"):
        generate.run_drift(path, samples_per_turn=3)


def test_model_generation_eos_ids_are_excluded_even_when_not_special(monkeypatch):
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(eos_token_id=[2, 99])
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(GenerationConfig=types.SimpleNamespace(from_pretrained=load)))
    ids = generate.load_token_exclusions({"repo_id": "test/model", "revision": "a" * 40}, types.SimpleNamespace(all_special_ids=[0, 2]))
    assert ids == [0, 2, 99]
    assert calls[0][1]["revision"] == "a" * 40
    row = {"response_token_ids": [8, 99, 9, 123], "excluded_token_ids": ids, "stop_reason": 123}
    assert run_state.content_positions(row) == [0, 2]


def test_vllm_requests_have_distinct_seeds_and_keep_actual_token_ids(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=lambda **kw: kw))
    jobs = [{"prompt_token_ids": [1, 23], "seed": seed} for seed in (4, 5, 6)]
    received = []
    def complete(prompts, params):
        received.append((prompts, params))
        # Text deliberately cannot determine these original token sequences.
        return [types.SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], outputs=[types.SimpleNamespace(
            text="same displayed text", token_ids=[80 + i, 99], finish_reason="stop", stop_reason=None,
        )]) for i, prompt in enumerate(prompts)]
    result = generate.sample_batch(types.SimpleNamespace(generate=complete), jobs, run_state.settings(3))
    assert [p["seed"] for p in received[0][1]] == [4, 5, 6]
    assert all(p["n"] == 1 for p in received[0][1])
    assert [r["response_token_ids"] for r in result] == [[80, 99], [81, 99], [82, 99]]
    assert all(r["prompt_token_ids"] == [1, 23] for r in result)


@pytest.mark.parametrize("mutation", ["seed", "sample_index", "missing_index", "prompt", "exclusions", "count", "duplicate"])
def test_invalid_saved_generation_samples_are_rejected(experiment, monkeypatch, mutation):
    path, _ = experiment
    generate.run_drift(path)
    state = run_state.read_record(path / "drift.state.json")
    state["status"] = "pending"
    run_state.save_record(path / "drift.state.json", state)
    rows = run_state.jsonl((path / "drift.jsonl").read_bytes())
    row = rows[0]
    if mutation == "seed": row["seed"] += 1
    elif mutation == "sample_index": row["sample_index"] = 9
    elif mutation == "missing_index": del row["sample_index"]
    elif mutation == "prompt": row["prompt_token_ids"][0] += 1
    elif mutation == "exclusions": row["excluded_token_ids"] = [999]
    elif mutation == "count": row["response_token_count"] = 999
    else: rows.append(dict(row))
    row["record_sha256"] = run_state.fingerprint({k: v for k, v in row.items() if k != "record_sha256"})
    (path / "drift.jsonl").write_bytes(b"".join(run_state.encoded(r) for r in rows))
    monkeypatch.setattr(generate, "load_drift_model", forbid)
    with pytest.raises(ValueError):
        generate.run_drift(path)


def test_completed_reader_rejects_missing_samples_and_stage_links(experiment, monkeypatch):
    path, _ = experiment
    run_both(path)
    monkeypatch.setattr(run_state, "environment", forbid)
    monkeypatch.setattr(run_state, "code_fingerprints", forbid)
    monkeypatch.setattr(run_state, "resolve_artifacts", forbid)
    assert len(run_state.load_completed_run(path)[3]) == 18
    state = run_state.read_record(path / "projections.state.json")
    state["identity"]["inputs"]["drift_sha256"] = "changed"
    state["stage_id"] = run_state.fingerprint(state["identity"])
    run_state.save_record(path / "projections.state.json", state)
    with pytest.raises(ValueError, match="stage does not match"):
        run_state.load_completed_run(path)


def test_completed_reader_requires_all_samples(experiment):
    path, _ = experiment
    run_both(path)
    rows = run_state.jsonl((path / "drift.jsonl").read_bytes())[:-1]
    state = run_state.read_record(path / "drift.state.json")
    run_state.complete_stage(path, "drift", state, rows)
    with pytest.raises(ValueError, match="missing planned responses"):
        run_state.load_completed_run(path)


def test_old_manifest_cannot_be_mixed_with_exact_token_runs(experiment, monkeypatch):
    path, _ = experiment
    generate.run_drift(path)
    run = run_state.read_record(path / "drift.manifest.json")
    run["schema_version"] = 1
    run_state.save_record(path / "drift.manifest.json", run)
    monkeypatch.setattr(generate, "load_drift_tokenizer", forbid)
    with pytest.raises(ValueError, match="predates sample/token"):
        generate.run_drift(path)


def test_exact_token_projector_matches_direct_tiny_llama_activations():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from assistant_axis.internals import ProbingModel

    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
    )).eval()
    pm = ProbingModel.from_existing(model, object(), model_name="tiny-llama")
    axis = torch.nn.functional.normalize(torch.randn(16), dim=0)
    cap_axis = torch.nn.functional.normalize(torch.randn(16), dim=0)
    row = {"prompt_token_ids": [1, 4, 5], "response_token_ids": [8, 2, 9, 3],
           "excluded_token_ids": [2, 3], "stop_reason": None, "reply": "Text is never tokenized."}
    captured = {}
    handles = [pm.get_layers()[i].register_forward_hook(
        lambda module, args, output, i=i: captured.__setitem__(i, (output[0] if isinstance(output, tuple) else output).detach())
    ) for i in (1, 2)]
    with torch.inference_mode():
        model(torch.tensor([row["prompt_token_ids"] + row["response_token_ids"]]), use_cache=False)
    for handle in handles:
        handle.remove()
    positions = [3, 5]
    expected = float(captured[1][0, positions].mean(0) @ axis)
    expected_fire = float((captured[2][0, positions] @ cap_axis > 0).float().mean())
    project = drift_activations.exact_token_projector(pm, axis, {2: (cap_axis, 0.0)}, 1)
    actual, fire = project(row)
    assert actual == pytest.approx(expected, abs=1e-6)
    assert fire == expected_fire
    assert all(not layer._forward_hooks for layer in pm.get_layers())
