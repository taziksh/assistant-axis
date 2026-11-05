"""Known-answer tests for repeated replies, paired changes, and plotted summaries."""

import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wild.drift import analysis
from wild.drift import state as run_state


def records(specification, samples=3):
    """Each dict entry is id: (domain, [per-turn lists of sample projections])."""
    conversations, generated, projections = [], [], []
    for cid, (domain, turns) in specification.items():
        conversations.append({"hash": cid, "domain": domain, "users": [f"user {k}" for k in range(len(turns))]})
        for k, values in enumerate(turns, 1):
            if values is None:
                continue
            assert len(values) == samples
            for index, value in enumerate(values):
                key = {"hash": cid, "k": k, "sample_index": index}
                generated.append({**key, "truncated": False})
                projections.append({**key, "proj": value})
    return conversations, generated, projections


def config(horizon=2, samples=3):
    return {"samples_per_turn": samples, "max_turns": horizon, "model": "test/llama", "layer": 40}


def point(summary, domain="coding", cohort="available", metric="projection", turn=1):
    return next(row for row in summary if (row["domain"], row["cohort"], row["metric"], row["turn"]) == (domain, cohort, metric, turn))


def test_three_samples_average_before_conversations_and_changes_are_paired():
    data = records({
        "a": ("coding", [[0, 1, 2], [3, 4, 5]]),
        "b": ("coding", [[8, 10, 12], [10, 12, 14]]),
    })
    turns = analysis.conversation_turns(*data, config())
    assert [row["projection"] for row in turns] == [1, 4, 10, 12]
    assert [row["change_from_turn1"] for row in turns] == [0, 3, 0, 2]
    summary, _ = analysis.summarize(data[0], turns, config())
    assert point(summary)["mean"] == 5.5
    assert point(summary, turn=2)["mean"] == 8
    assert point(summary, metric="change", turn=2)["mean"] == 2.5
    assert point(summary)["n_conversations"] == 2


def test_null_sample_excludes_whole_turn_and_missing_baseline_excludes_change():
    data = records({"a": ("coding", [[1, None, 3], [7, 8, 9]])})
    turns = analysis.conversation_turns(*data, config())
    assert turns[0]["status"] == "no_content_tokens"
    assert turns[0]["valid_samples"] == 2
    assert turns[0]["projection"] is None
    assert turns[1]["projection"] == 8
    assert turns[1]["change_from_turn1"] is None
    summary, _ = analysis.summarize(data[0], turns, config())
    assert point(summary, turn=2)["n_conversations"] == 1
    assert point(summary, metric="change", turn=2)["n_conversations"] == 0


@pytest.mark.parametrize("mutation", ["duplicate", "missing_projection", "missing_sample", "unexpected_sample", "infinite"])
def test_inconsistent_saved_samples_fail_instead_of_disappearing(mutation):
    data = records({"a": ("coding", [[1, 2, 3]])})
    conversations, generated, projections = copy.deepcopy(data)
    if mutation == "duplicate":
        projections.append(projections[0].copy())
    elif mutation == "missing_projection":
        projections.pop()
    elif mutation == "missing_sample":
        generated.pop()
        projections.pop()
    elif mutation == "unexpected_sample":
        generated[0]["sample_index"] = projections[0]["sample_index"] = 9
    else:
        projections[0]["proj"] = float("inf")
    with pytest.raises(ValueError):
        analysis.conversation_turns(conversations, generated, projections, config())


def test_attrition_does_not_become_a_false_paired_change():
    data = records({
        "short": ("coding", [[100, 100, 100]]),
        "long": ("coding", [[10, 10, 10], [11, 11, 11]]),
    })
    turns = analysis.conversation_turns(*data, config())
    summary, cohorts = analysis.summarize(data[0], turns, config())
    assert point(summary)["mean"] == 55
    assert point(summary, turn=2)["mean"] == 11
    assert point(summary, metric="change", turn=2)["mean"] == 1
    assert cohorts["coding"] == ["long"]
    assert point(summary, cohort="fixed")["mean"] == 10
    assert point(summary, cohort="fixed", turn=2)["mean"] == 11
    assert point(summary, cohort="fixed")["ci_low"] is None


def test_ungenerated_turns_and_truncation_are_counted():
    data = records({"a": ("coding", [[1, 2, 3], None])})
    data[1][0]["truncated"] = True
    turns = analysis.conversation_turns(*data, config())
    assert turns[0]["truncated_samples"] == 1
    assert turns[0]["projection"] == 2
    assert turns[1]["status"] == "not_generated"
    assert turns[1]["generated_samples"] == 0
    summary, cohorts = analysis.summarize(data[0], turns, config())
    assert cohorts["coding"] == []
    assert point(summary, turn=2)["n_conversations"] == 0


def test_bootstrap_resamples_whole_trajectories_and_is_deterministic():
    values = np.array([[0, 100], [2, 4], [8, np.nan]])
    draws, seed = 31, 17
    rng = np.random.default_rng(seed)
    picked = [values[rng.integers(0, len(values), len(values))] for _ in range(draws)]
    result = analysis.trajectory_summary(values, draws, seed)
    assert analysis.trajectory_summary(values, draws, seed) == result
    for column in range(2):
        means = [np.mean(matrix[:, column][np.isfinite(matrix[:, column])])
                 for matrix in picked if np.isfinite(matrix[:, column]).any()]
        low, high = np.percentile(means, [2.5, 97.5])
        assert result[column]["ci_low"] == low
        assert result[column]["ci_high"] == high
        assert result[column]["bootstrap_valid_draws"] == len(means)


def test_absent_categories_and_single_conversations_do_not_get_fake_intervals():
    data = records({"a": ("coding", [[1, 2, 3]])})
    turns = analysis.conversation_turns(*data, config())
    summary, cohorts = analysis.summarize(data[0], turns, config())
    assert len(summary) == 4 * 2 * 2 * 2
    assert point(summary)["mean"] == 2
    assert point(summary)["ci_low"] is None
    assert point(summary, domain="ai")["mean"] is None
    assert point(summary, domain="ai")["n_conversations"] == 0
    assert cohorts == dict.fromkeys(analysis.DOMAINS, [])


def test_adding_more_replies_does_not_add_independent_conversations():
    data = records({"a": ("coding", [[0, 2]]), "b": ("coding", [[8, 10]])}, samples=2)
    settings = config(horizon=1, samples=2)
    summary, _ = analysis.summarize(data[0], analysis.conversation_turns(*data, settings), settings)
    assert point(summary)["mean"] == 5
    assert point(summary)["n_conversations"] == 2
    assert (point(summary)["ci_low"], point(summary)["ci_high"]) == (1, 9)


def test_run_writes_actual_plot_files_tables_and_honest_report(tmp_path, monkeypatch):
    data = records({
        "code": ("coding", [[1, 2, 3], [2, 3, 4]]),
        "write": ("writing", [[2, 3, 4]]),
        "therapy": ("therapy", [[5, 6, 7], [3, 4, 5]]),
        "ai": ("ai", [[5, 6, 7], [4, None, 6]]),
    })
    run = {"run_id": "synthetic-test-only", "config": config()}
    (tmp_path / "selected.manifest.json").write_text(json.dumps({"review_status": "unreviewed"}))
    (tmp_path / "projections.jsonl").write_bytes(b"".join(run_state.encoded(row) for row in data[2]))
    monkeypatch.setattr(run_state, "load_completed_run", lambda directory: (data[0], run, data[1], data[2]), raising=False)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    result = analysis.run_analysis(tmp_path)
    assert result["review_status"] == "unreviewed"
    assert result["counts"]["ai"]["unmeasurable_turns"] == 1
    assert result["fixed_cohort_ids"]["writing"] == []
    output = tmp_path / "analysis"
    for name in ("projection_by_turn", "change_from_first_turn", "conversation_counts", "fixed_cohort"):
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert "<svg" in (output / f"{name}.svg").read_text()
    rows = list(csv.DictReader((output / "conversation_turns.csv").open()))
    assert len(rows) == 7
    assert {row["domain"] for row in rows} == set(analysis.DOMAINS)
    assert json.loads((output / "results.json").read_text()) == result
    report = (output / "REPORT.md").read_text()
    assert "unreviewed" in report
    assert "not describe an autonomous Llama conversation" in report
    assert "pointwise intervals" in report


def test_incomplete_run_fails_before_creating_analysis_outputs(tmp_path, monkeypatch):
    def incomplete(directory):
        raise ValueError("generation is incomplete")
    monkeypatch.setattr(run_state, "load_completed_run", incomplete, raising=False)
    with pytest.raises(ValueError, match="incomplete"):
        analysis.run_analysis(tmp_path)
    assert not (tmp_path / "analysis").exists()
