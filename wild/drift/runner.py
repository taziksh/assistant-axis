"""Generate replies, measure projections, and plot the four conversation domains."""

import argparse
from pathlib import Path
import subprocess
import sys

from wild.common.paths import REPO_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--model-revision")
    parser.add_argument("--axis-revision")
    parser.add_argument("--samples-per-turn", type=int, help="replies per turn (new runs default to 3; resumes reuse the saved value)")
    args = parser.parse_args(argv)
    if args.samples_per_turn is not None and args.samples_per_turn < 1:
        parser.error("samples-per-turn must be positive")
    run_dir = str(Path(args.run_dir).resolve())
    command = [sys.executable, "-m", "wild.drift.generation", run_dir]
    if args.model_revision:
        command += ["--model-revision", args.model_revision]
    if args.axis_revision:
        command += ["--axis-revision", args.axis_revision]
    if args.samples_per_turn is not None:
        command += ["--samples-per-turn", str(args.samples_per_turn)]
    # Separate processes release vLLM's workers and GPU memory before HF loads.
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    subprocess.run([sys.executable, "-m", "wild.drift.measurement", run_dir], check=True, cwd=REPO_ROOT)
    subprocess.run([sys.executable, "-m", "wild.drift.analysis", run_dir], check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
