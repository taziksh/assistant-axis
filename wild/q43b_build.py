"""Build the Section 4.3 rerun on the paper's Qwen openings with the paper's own
kind of second-turn questions.

Usage: q43b_build.py OUT_DIR [--first-turn-questions N]
The paper asked 440 harmful questions "from the jailbreak dataset" after each of
its 2,750 role openings. Which 440 is not stated; its example question is in the
released persona-jailbreak set (data/lu, see its PROVENANCE.md). We draw 10
questions per harm category from jailbreak_1100.jsonl (the set its capping
evaluation used): 2 from each of the category's 5 personas, seeded. The category
in lu_prep.EXCLUDED is left out, which leaves 430 questions.
Openings: the paper crossed 275 roles with 10 first-turn questions. We keep the
openings of N of those 10 questions (seeded, the same N for every role; default
2, i.e. 550 openings). A later run with a larger N adds openings; harm_wild.py
--chunk skips replies already saved.
Reads results/q43/{selected,drift,projections}.jsonl (the openings, the
authors' own Qwen replies and our Assistant Axis projection of each) and
results/q43/paper_proj.jsonl (which first-turn question each opening used).
Writes OUT_DIR/questions.jsonl and OUT_DIR/{selected,drift,projections}.jsonl.
"""

import argparse
import collections
import json
import os
import random

from lu_prep import EXCLUDED

HERE = os.path.dirname(os.path.abspath(__file__))
PER_PERSONA = 2


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--first-turn-questions", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    by_persona = collections.defaultdict(list)
    for r in read(os.path.join(HERE, "data", "lu", "jailbreak_1100.jsonl")):
        if r["harm_category"] not in EXCLUDED:
            by_persona[(r["harm_category"], r["role"])].append(r)
    rng = random.Random(0)
    questions = []
    for key in sorted(by_persona):
        for r in rng.sample(by_persona[key], PER_PERSONA):
            questions.append({
                "id": f"jb{r['id']}",
                "category": r["harm_category"],
                "persona": r["persona"],
                "question": r["question"],
            })
    write(os.path.join(args.out_dir, "questions.jsonl"), questions)
    per_category = collections.Counter(q["category"] for q in questions)
    print(
        f"questions: {len(questions)} in {len(per_category)} categories"
        f" ({set(per_category.values())} per category)"
    )

    src = os.path.join(HERE, "results", "q43")
    index = {
        r["hash"]: r["question_index"]
        for r in read(os.path.join(src, "paper_proj.jsonl"))
    }
    # Always shuffle all 10, then keep a prefix: a larger N is a superset.
    order = sorted(set(index.values()))
    random.Random(0).shuffle(order)
    keep = set(order[: args.first_turn_questions])
    for name in ("selected", "drift", "projections"):
        rows = [
            r
            for r in read(os.path.join(src, f"{name}.jsonl"))
            if index[r["hash"]] in keep
        ]
        write(os.path.join(args.out_dir, f"{name}.jsonl"), rows)
    print(f"openings: {len(rows)} (first-turn questions {sorted(keep)})")


if __name__ == "__main__":
    main()
