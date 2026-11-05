"""Keep repeated replies out of the historical single-response analysis."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import analyze
from wild.drift import analysis as drift_analysis
from wild.drift import state as run_state


def test_legacy_drift_rejects_repeated_reply_format(tmp_path):
    (tmp_path / "projections.jsonl").write_text(json.dumps({"hash": "chat", "k": 1, "sample_index": 0, "proj": 1.0}) + "\n")
    with pytest.raises(ValueError, match="drift_analysis.py"):
        analyze.drift(str(tmp_path), [], {}, [], True)


def test_modern_runs_use_sample_aware_analysis(tmp_path, monkeypatch):
    (tmp_path / "drift.manifest.json").write_text('{"schema_version": 2}')
    (tmp_path / "projections.jsonl").write_text("")
    run_state.save_record(tmp_path / "projections.state.json", {"status": "complete"})
    calls = []
    monkeypatch.setattr(drift_analysis, "main", lambda args: calls.append(args))
    analyze.main([str(tmp_path)])
    assert calls == [[str(tmp_path)]]


def test_modern_partial_extraction_defers_early_legacy_analysis(tmp_path, monkeypatch, capsys):
    (tmp_path / "drift.manifest.json").write_text('{"schema_version": 2}')
    (tmp_path / "projections.jsonl").write_text('{"sample_index": 0}\n')
    run_state.save_record(tmp_path / "projections.state.json", {"status": "pending"})
    monkeypatch.setattr(drift_analysis, "main", lambda args: pytest.fail("partial results must not be analyzed"))
    monkeypatch.setattr(analyze, "drift", lambda *args: pytest.fail("legacy analysis would lose reply samples"))
    analyze.main([str(tmp_path)])
    assert "deferred" in capsys.readouterr().out
    assert not (tmp_path / "REPORT.md").exists()
