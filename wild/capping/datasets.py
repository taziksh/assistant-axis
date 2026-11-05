"""Jailbreak sets for the capping and harm experiments (shared by capping.py and
judge_paper.py).

- itw: in-the-wild jailbreak prompts (TrustAIRLab, 2023-12-25) paired with
  forbidden questions (TrustAIRLab): N distinct questions from each of the 13
  categories, each with a random prompt.
- mhj: Scale AI's multi-turn human jailbreaks (537); behavior = HarmBench row
  question_id - 2. (ToxicChat's jailbreak-flagged prompts were dropped
  2026-11-02: no target behavior for the judge.)
"""

import csv
import json
import os
import random

import pandas as pd

SLOT = "[INSERT PROMPT HERE]"


def load_sets(dl, n_itw, itw_templates=1):
    hb = list(
        csv.DictReader(
            open(os.path.join(dl, "harmbench_behaviors_text_all.csv"))
        )
    )
    rng = random.Random(0)
    j = pd.read_parquet(
        os.path.join(
            dl, "itw", "jailbreak_2023_12_25", "train-00000-of-00001.parquet"
        )
    )
    q = pd.read_csv(os.path.join(dl, "fq", "forbidden_question_set.csv"))
    # n_itw per content-policy category: distinct questions, each with
    # itw_templates random jailbreak prompts (ids gain a -t suffix when > 1).
    itw = []
    for cat, qs in q.groupby("content_policy_name", sort=True):
        for qq in qs.sample(n=min(n_itw, len(qs)), random_state=0).itertuples():
            for t in range(itw_templates):
                p = j.prompt.iloc[rng.randrange(len(j))]
                text = (
                    p.replace(SLOT, qq.question)
                    if SLOT in p
                    else f"{p}\n\n{qq.question}"
                )
                itw.append({
                    "id": (
                        f"itw{qq.Index}"
                        + (f"-{t}" if itw_templates > 1 else "")
                    ),
                    "turns": [text],
                    "system": None,
                    "behavior": qq.question,
                    "context": None,
                    "category": cat,
                })
    m = pd.read_csv(os.path.join(dl, "mhj", "harmbench_behaviors.csv"))
    mhj = []
    for i, r in m.iterrows():
        msgs = [
            json.loads(r[c])
            for c in m.columns
            if c.startswith("message_") and isinstance(r[c], str)
        ]
        b = hb[int(r.question_id) - 2]
        mhj.append({
            "id": f"mhj{i}",
            "system": next(
                (x["body"] for x in msgs if x["role"] == "system"), None
            ),
            "turns": [x["body"] for x in msgs if x["role"] == "user"],
            "behavior": b["Behavior"],
            "context": b["ContextString"] or None,
            "category": b["SemanticCategory"],
            "tactic": r.tactic,
        })
    return {"itw": itw, "mhj": mhj}
