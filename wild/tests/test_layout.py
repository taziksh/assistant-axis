"""Check command routing and repository assets after moving experiment files."""

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wild import __main__ as commands
from wild.common.paths import REPO_ROOT, WILD_ROOT
from wild.drift import runner
from wild.harm import roles
from wild.legacy import calibration


def test_every_command_target_is_importable_without_loading_models():
    for module in commands.COMMANDS.values():
        spec = importlib.util.find_spec(module)
        assert spec is not None and Path(spec.origin).is_file(), module


def test_dispatcher_preserves_arguments_and_restores_argv(monkeypatch):
    calls = []
    before = sys.argv
    monkeypatch.setattr(commands.runpy, "run_module", lambda module, **kwargs: calls.append((module, list(sys.argv), kwargs)))
    commands.main(["judge.py", "candidates with spaces.jsonl", "run", "--per-domain", "10"])
    assert calls == [("wild.selection.judge", ["wild.selection.judge", "candidates with spaces.jsonl", "run", "--per-domain", "10"], {"run_name": "__main__", "alter_sys": True})]
    assert sys.argv is before
    with pytest.raises(SystemExit) as error:
        commands.main(["unknown-command"])
    assert error.value.code == 2


def test_runner_preserves_relative_output_location(monkeypatch, tmp_path):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    runner.main(["relative-run", "--samples-per-turn", "3"])
    assert len(calls) == 3
    assert all(call[0][3] == str(tmp_path / "relative-run") for call in calls)
    assert all(call[1] == {"check": True, "cwd": REPO_ROOT} for call in calls)
    assert calls[0][0][-2:] == ["--samples-per-turn", "3"]


def test_role_openings_match_the_pre_move_output(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["roles", str(tmp_path), "--no-role", "--questions", "0", "2"])
    roles.main()
    # Captured from the original roles_build.py against the same released assets.
    assert hashlib.sha256((tmp_path / "selected.jsonl").read_bytes()).hexdigest() == "db59acde9f37c6138e93f8bcbcf796693fa4ffa26a197b42dcb9c85a23ec9040"


def test_calibration_reads_the_same_released_transcripts(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["calibration", str(tmp_path)])
    calibration.main()
    expected = {
        "selected.jsonl": "f4c587e539aec508a1d86e5d485cb3bddbef14fb391e0d7e8c600d7bb704eff3",
        "drift.jsonl": "3cd7dab74551e54265bb4947e220e89c9aa05bbd64c4fbaafa010677aa702631",
    }
    for filename, digest in expected.items():
        assert hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() == digest


def test_shell_workflows_use_existing_modules():
    import re

    for path in (WILD_ROOT / "scripts").glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
        modules = re.findall(r"\$PY\s+-m\s+(wild\.[\w.]+)", path.read_text())
        assert modules, path
        for module in modules:
            assert importlib.util.find_spec(module) is not None, (path, module)
