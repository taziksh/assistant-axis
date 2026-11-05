"""Run only the protected generation and projection stages, without remote lifecycle actions."""

import argparse
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--model-revision")
    parser.add_argument("--axis-revision")
    args = parser.parse_args(argv)
    here = Path(__file__).resolve().parent
    command = [sys.executable, str(here / "generate.py"), args.run_dir, "--drift-only"]
    if args.model_revision:
        command += ["--model-revision", args.model_revision]
    if args.axis_revision:
        command += ["--axis-revision", args.axis_revision]
    # Separate processes release vLLM's workers and GPU memory before HF loads.
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(here / "drift_activations.py"), args.run_dir], check=True)


if __name__ == "__main__":
    main()
