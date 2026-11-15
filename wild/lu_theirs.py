"""Split the output of Lu et al.'s capping script into judge_paper.py inputs.

Usage: lu_theirs.py THEIRS_JSONL OUT_DIR
Writes OUT_DIR/{experiment_id}.jsonl (":" replaced by "_"), one row per reply,
with the same id, persona, category and judge request as lu_prep.py. Reports
prompts with no reply (their script drops empty replies).
"""

import collections
import json
import os
import sys


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    by_exp = collections.defaultdict(list)
    with open(src) as f:
        for line in f:
            r = json.loads(line)
            by_exp[r["experiment_id"]].append({
                "id": f"lu{r['id']}",
                "persona": r["persona"],
                "category": r["harm_category"],
                "request": r["prompt"] + " " + r["question"],
                "replies": [r["response"]],
            })
    for exp, rows in by_exp.items():
        path = os.path.join(out, exp.replace(":", "_") + ".jsonl")
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(f"{exp}: {len(rows)} replies ({1075 - len(rows)} missing) -> {path}")


if __name__ == "__main__":
    main()
