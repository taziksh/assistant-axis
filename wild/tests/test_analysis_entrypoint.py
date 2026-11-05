"""Keep repeated replies out of the historical single-response analysis."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import analyze
from wild.drift import state as run_state


def test_legacy_drift_rejects_repeated_reply_format(tmp_path):
    (tmp_path / "projections.jsonl").write_text(json.dumps({"hash": "chat", "k": 1, "sample_index": 0, "proj": 1.0}) + "\n")
    with pytest.raises(ValueError, match="drift_analysis.py"):
        analyze.drift(str(tmp_path), [], {}, [], True)


