"""Exercise the selection/projection handoff using synthetic data and fake models."""

import argparse
import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import build
from wild.drift import measurement as drift_activations
from wild.drift import analysis as drift_analysis
from wild.drift import generation as generate
import judge
from wild.drift import state as run_state
import selection


def test_four_domain_selection_handoff_and_verified_reuse(tmp_path, monkeypatch):
    revision = "a" * 40
    candidates = []
    for domain in selection.DOMAINS:
        source = {
            "conversation_hash": domain, "language": "English", "model": "fixture",
            "conversation": [
                {"role": role, "content": f"{domain} {role} {turn}"}
                for turn in range(3) for role in ("user", "assistant")
            ],
        }
        candidate, reason = build.candidate(source, "fixture.parquet", len(candidates), 3, 3)
        assert reason is None
        candidates.append(candidate)

    hub = types.SimpleNamespace(dataset_info=lambda *a, **kw: types.SimpleNamespace(sha=revision))
    fs = types.SimpleNamespace(glob=lambda pattern: [pattern.replace("*", "fixture")])
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=lambda: hub, HfFileSystem=lambda: fs))
    monkeypatch.setattr(build, "scan", lambda *a: (candidates, {}))
    pool = tmp_path / "candidates.jsonl"
    build.run(argparse.Namespace(
        out=str(pool), dataset=build.DATASET, revision="main", min_turns=3,
        max_turns=3, others=10, random=None, max_files=None, seed=5,
    ))

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            text = json.dumps(messages)
            return list(text.encode()) if kwargs.get("tokenize", True) else text

        def __call__(self, text, **kwargs):
            return types.SimpleNamespace(input_ids=list(text.encode()))

    def classify(prompts):
        answers = []
        for prompt in prompts:
            messages = json.loads(bytes(prompt))
            users = json.loads(messages[1]["content"])["user_messages"]
            answers.append(json.dumps({"domain": users[0].split()[0], "rationale": "Synthetic topic fixture."}))
        return answers

    output = tmp_path / "run"
    args = argparse.Namespace(
        convs=str(pool), run_dir=str(output), per_domain=1, min_turns=3,
        max_turns=3, seed=5, model_revision="main", context_limit=16384, output_tokens=256,
    )
    result = judge.run(args, model_loader=lambda *a: (Tokenizer(), classify), revision_resolver=lambda *a: revision)
    assert result["counts"]["selected_counts"] == dict.fromkeys(selection.DOMAINS, 1)
    records, provenance, _ = run_state.selection(output)
    assert provenance["dataset"]["revision"] == revision
    assert provenance["review_status"] == "unreviewed"
    assert {row["domain"] for row in records} == set(selection.DOMAINS)

    artifacts = {
        "model": {"repo_id": "fixture/model", "revision": revision},
        **{name: {"repo_id": "fixture/vectors", "revision": revision,
                  "filename": name + ".pt", "sha256": "b" * 64}
           for name in ("axis", "capping")},
    }
    monkeypatch.setattr(run_state, "resolve_artifacts", lambda *a: artifacts)
    monkeypatch.setattr(run_state, "environment", lambda stage: {"fixture": "1"})
    monkeypatch.setattr(generate, "load_drift_tokenizer", lambda *a: Tokenizer())
    monkeypatch.setattr(generate, "load_token_exclusions", lambda *a: [0])
    monkeypatch.setattr(generate, "drift_runtime", lambda: {"tensor_parallel_size": 1, "gpus": ["fixture"]})
    monkeypatch.setattr(generate, "load_drift_model", lambda *a: object())
    monkeypatch.setattr(generate, "sample_batch", lambda model, jobs, config: [
        {"prompt_token_ids": job["prompt_token_ids"], "response_token_ids": [7, 8, 0],
         "reply": "Synthetic reply", "finish": "stop", "stop_reason": 0}
        for job in jobs
    ])
    monkeypatch.setattr(drift_activations, "projection_runtime", lambda: {"gpus": ["fixture"]})
    monkeypatch.setattr(drift_activations, "load_projector", lambda *a: lambda row: (float(row["k"] + row["sample_index"]), 0.0))
    generate.main([str(output)])
    drift_activations.main([str(output)])
    projections = selection.read_jsonl(output / "projections.jsonl")
    assert {(row["hash"], row["k"], row["sample_index"]) for row in projections} == {
        (domain, turn, sample) for domain in selection.DOMAINS for turn in (1, 2, 3) for sample in range(3)
    }
    assert len(projections) == 36
    assert all(row["response_token_count"] == 2 for row in projections)

    def forbidden(*args, **kwargs):
        raise AssertionError("verified cache attempted model loading or Hub access")

    monkeypatch.setattr(generate, "load_drift_tokenizer", forbidden)
    monkeypatch.setattr(drift_activations, "load_projector", forbidden)
    monkeypatch.setattr(run_state, "resolve_artifacts", forbidden)
    judge.run(args, model_loader=forbidden, revision_resolver=forbidden)
    generate.main([str(output)])
    drift_activations.main([str(output)])
    drift_analysis.main([str(output)])
    assert (output / "analysis" / "projection_by_turn.png").is_file()
