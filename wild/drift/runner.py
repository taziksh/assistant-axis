"""Generate replies, measure projections, and plot the four conversation domains."""

import argparse
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--model-revision")
    parser.add_argument("--axis-revision")
    parser.add_argument("--samples-per-turn", type=int, help="replies per turn (new runs default to 3; resumes reuse the saved value)")
    args = parser.parse_args(argv)
    if args.samples_per_turn is not None and args.samples_per_turn < 1:
        parser.error("samples-per-turn must be positive")
    here = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(here / "generate.py"), args.run_dir, "--drift-only"]
    if args.model_revision:
        command += ["--model-revision", args.model_revision]
    if args.axis_revision:
        command += ["--axis-revision", args.axis_revision]
    if args.samples_per_turn is not None:
        command += ["--samples-per-turn", str(args.samples_per_turn)]
    # Separate processes release vLLM's workers and GPU memory before HF loads.
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(here / "drift_activations.py"), args.run_dir], check=True)
    subprocess.run([sys.executable, str(here / "drift_analysis.py"), args.run_dir], check=True)


if __name__ == "__main__":
    main()
