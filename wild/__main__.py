"""Run an experiment using its familiar command name."""

import argparse
import runpy
import sys

COMMANDS = {
    "build": "wild.selection.build",
    "judge": "wild.selection.judge",
    "drift_run": "wild.drift.runner",
    "drift_activations": "wild.drift.measurement",
    "drift_analysis": "wild.drift.analysis",
    "harm_wild": "wild.harm.generate",
    "alone": "wild.harm.baseline",
    "roles_build": "wild.harm.roles",
    "judge_harm": "wild.harm.judge",
    "judge_paper": "wild.harm.paper_judge",
    "classify": "wild.harm.classifiers",
    "shah_gen": "wild.harm.shah",
    "capping": "wild.capping.evaluate",
    "capability": "wild.capping.capability",
    "casestudy": "wild.capping.case_studies",
    "analyze": "wild.reports.analyze",
    "tables": "wild.reports.tables",
    "figures": "wild.reports.figures",
    "ingest_docent": "wild.reports.docent",
    "generate": "wild.legacy.generation",
    "activations": "wild.legacy.measurement",
    "drift_scale": "wild.legacy.calibration",
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run a real-chat experiment. Arguments after the command are passed through unchanged.",
        epilog="Commands: " + ", ".join(COMMANDS),
    )
    parser.add_argument("command", nargs="?")
    if not args or args[0] in ("-h", "--help"):
        parser.print_help()
        return
    command = args.pop(0).removesuffix(".py")
    if command not in COMMANDS:
        parser.error(f"unknown command: {command}")
    previous = sys.argv
    try:
        sys.argv = [COMMANDS[command], *args]
        runpy.run_module(COMMANDS[command], run_name="__main__", alter_sys=True)
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
