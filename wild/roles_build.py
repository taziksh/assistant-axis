"""Reproduce the paper's Section 4.3 openings: role system prompt + an unrelated
question (first turn).

Usage: roles_build.py RUN_DIR [--questions FROM TO] [--no-role]
--questions 0 2 builds 275 roles x the first 2 of the 10 questions (550
openings; the first stage, 2026-11-03).
--no-role builds the no-persona baseline: the same 10 questions with no system
prompt.
The paper combined 275 role system prompts with 10 questions (2,750 first
turns), then asked one harmful
question per second turn. It released the role prompts (data/roles/instructions,
276 files: the 275 roles
plus "default", left out here) and its extraction questions
(data/extraction_questions.jsonl, 240), but
not which instruction variant or which 10 questions it used. We use one random
variant per opening and
the same 10 random extraction questions for every role (both seeded).
Writes RUN_DIR/selected.jsonl in the format harm_wild.py and activations.py read
(one user turn, a
system prompt), so the rest of the pipeline (pool -> activations -> harm) is
unchanged.
"""

import argparse
import glob
import json
import os
import random

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
N_QUESTIONS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--questions", nargs=2, type=int, default=[0, N_QUESTIONS])
    ap.add_argument(
        "--no-role",
        action="store_true",
        help="the same questions with no system prompt (baseline)",
    )
    args = ap.parse_args()
    qs = [
        json.loads(l)
        for l in open(os.path.join(ROOT, "extraction_questions.jsonl"))
    ]
    questions = random.Random(0).sample(qs, N_QUESTIONS)
    rows = []
    if args.no_role:
        rows = [
            {
                "hash": f"none-q{q['id']}",
                "role": "none",
                "system": None,
                "users": [q["question"]],
                "gpt": [],
            }
            for q in questions[args.questions[0] : args.questions[1]]
        ]
    for path in (
        []
        if args.no_role
        else sorted(
            glob.glob(os.path.join(ROOT, "roles", "instructions", "*.json"))
        )
    ):
        role = os.path.basename(path)[:-5]
        if role == "default":
            continue
        variants = [v["pos"] for v in json.load(open(path))["instruction"]]
        rng = random.Random(role)
        for q in questions[args.questions[0] : args.questions[1]]:
            rows.append({
                "hash": f"{role}-q{q['id']}",
                "role": role,
                "system": rng.choice(variants),
                "users": [q["question"]],
                "gpt": [],
            })
    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "selected.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = args.questions[1] - args.questions[0]
    print(
        f"{len(rows)} openings from {len(rows) // n}"
        f" {'no-role' if args.no_role else 'roles'} x {n} questions"
    )


if __name__ == "__main__":
    main()
