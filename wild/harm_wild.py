"""Harm after drift on real chats (paper Section 4.3), with real WildChat
openings instead of role prompts.

Usage: harm_wild.py RUN_DIR CATQA_JSONL {pool,harm} [--first-max-tokens N]
       [--per-band N] [--per-category Q]
       [--lowest N] [--only-k K] [--hash-regex RE] [--questions-jsonl FILE]
       [--tag T]
Design (approved 2026-11-03), following the paper:
- pool: for each conversation in RUN_DIR/selected.jsonl (real WildChat chats
  from build.py --random, or the
  paper's role openings from roles_build.py), pick a random user turn k (1..12,
  prompt must fit) and let
  Llama write reply k after the history. Writes RUN_DIR/drift.jsonl, which
  activations.py projects (the
  opening's score = the paper's first-turn projection).
- harm: openings are chosen by 10 equal-width bands of the opening score (N per
  band; a large N takes all),
  or the N most drifted (--lowest), optionally only turn K (--only-k) or hashes
  matching --hash-regex
  (needs RUN_DIR/projections.jsonl).
  Every opening is followed by the same harmful questions (the paper used the
  same 440 for every
  opening). The paper's set (Shah et al. 2023) was withheld by its authors, so
  we use the closest
  published set in style, CategoricalHarmfulQA (Bhardwaj et al. 2024, English,
  Apache-2.0): open-ended
  requests for strategies. Q per category, fixed seed. The "Child Abuse"
  category is excluded so that
  no replies about harming children are generated or stored. --questions-jsonl
  uses another set instead
  (e.g. the Shah-style questions). One reply per opening x question.
  Writes RUN_DIR/harm_openings{T}.jsonl and RUN_DIR/harm_wild{T}.jsonl.
Generation: vLLM, temperature 0.7, top_p 0.9 (the repo's settings), no system
            prompt (except role openings from
roles_build.py, which carry the role prompt as in the paper).
"""

import argparse
from collections import defaultdict
import os
import random
import re

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM
from vllm import SamplingParams
from wcommon import HARM_NEW_TOKENS
from wcommon import history
from wcommon import MAX_MODEL_LEN
from wcommon import MAX_NEW_TOKENS
from wcommon import MAX_PROMPT_TOKENS
from wcommon import MAX_TURN
from wcommon import MODEL
from wcommon import prompt_ids
from wcommon import read_jsonl
from wcommon import TEMPERATURE
from wcommon import TOP_P
from wcommon import write_jsonl

N_BANDS = 10


EXCLUDED = {"Child Abuse"}


def pick_questions(path, per_category):
    """per_category CategoricalHarmfulQA questions from each category except
    EXCLUDED (seeded)."""
    import json

    rows = [json.loads(l) for l in open(path) if l.strip()]
    by_cat = defaultdict(list)
    for i, r in enumerate(rows):
        if r["Category"] not in EXCLUDED:
            by_cat[r["Category"]].append({
                "id": f"catqa{i}",
                "category": r["Category"],
                "subcategory": r["Subcategory"],
                "question": r["Question"],
            })
    if per_category > 4:
        raise ValueError(
            "per_category is at most 4: smaller sets are prefixes of the 4"
            " drawn per category"
        )
    rng = random.Random(0)
    out = []
    for c in sorted(
        by_cat
    # Always draw 4, then keep the first per_category: smaller sets are subsets.
    ):
        out += rng.sample(by_cat[c], 4)[:per_category]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("catqa_jsonl")
    ap.add_argument("stage", choices=["pool", "harm"])
    ap.add_argument("--per-band", type=int, default=30)
    ap.add_argument("--per-category", type=int, default=4)
    ap.add_argument(
        "--max-openings",
        type=int,
        default=None,
        help="cap on openings (the quick floor check)",
    )
    ap.add_argument(
        "--lowest",
        type=int,
        default=None,
        help=(
            "instead of bands: the N most drifted openings not already in"
            " harm_openings.jsonl"
        ),
    )
    ap.add_argument(
        "--only-k",
        type=int,
        default=None,
        help="only openings at this user turn (e.g. 1)",
    )
    ap.add_argument(
        "--questions-jsonl",
        default=None,
        help=(
            "use these harmful questions instead of CategoricalHarmfulQA (rows:"
            " id, category, question)"
        ),
    )
    ap.add_argument(
        "--first-max-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=(
            "reply limit for the opening turn (pool); the repo's role rollouts"
            " used 512"
        ),
    )
    ap.add_argument(
        "--hash-regex",
        default=None,
        help=(
            "only openings whose hash matches (e.g. a subset of role openings)"
        ),
    )
    ap.add_argument(
        "--keep-truncated",
        action="store_true",
        help=(
            "also use openings whose reply hit the length limit (role openings:"
            " the paper's code keeps them)"
        ),
    )
    ap.add_argument(
        "--tag",
        default="",
        help="output suffix, so a second harm run doesn't overwrite the first",
    )
    args = ap.parse_args()
    convs = {
        c["hash"]: c
        for c in read_jsonl(os.path.join(args.run_dir, "selected.jsonl"))
    }
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        MODEL,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.92,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True,
    )
    params = lambda n: SamplingParams(
        temperature=TEMPERATURE, top_p=TOP_P, max_tokens=n, seed=0
    )

    if args.stage == "pool":
        units, prompts = [], []
        for h, c in convs.items():
            rng = random.Random(h)
            ks = list(range(1, min(len(c["users"]), MAX_TURN) + 1))
            rng.shuffle(ks)
            for k in ks:  # A random turn whose prompt fits.
                p = prompt_ids(
                    tok, history(c["users"], c["gpt"], k, c.get("system"))
                )
                if len(p) <= MAX_PROMPT_TOKENS:
                    units.append((c, k))
                    prompts.append(p)
                    break
        outs = llm.generate(
            [{"prompt_token_ids": p} for p in prompts],
            params(args.first_max_tokens),
        )
        write_jsonl(
            os.path.join(args.run_dir, "drift.jsonl"),
            [
                {
                    "hash": c["hash"],
                    "domain": "random",
                    "k": k,
                    "reply": o.outputs[0].text,
                    "finish": o.outputs[0].finish_reason,
                }
                for (c, k), o in zip(units, outs)
            ],
        )
        print(f"pool: {len(units)} openings", flush=True)
        return

    proj = {
        (r["hash"], r["k"]): r["proj"]
        for r in read_jsonl(os.path.join(args.run_dir, "projections.jsonl"))
        if r["proj"] is not None
    }
    openings = [
        r
        for r in read_jsonl(os.path.join(args.run_dir, "drift.jsonl"))
        if (r["hash"], r["k"]) in proj
        and (r["finish"] == "stop" or args.keep_truncated)
        and args.only_k in (None, r["k"])
        and (args.hash_regex is None or re.search(args.hash_regex, r["hash"]))
    ]
    x = np.array([proj[(r["hash"], r["k"])] for r in openings])
    edges = np.linspace(x.min(), x.max(), N_BANDS + 1)
    band = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, N_BANDS - 1)
    rng = random.Random(0)
    chosen = []
    if args.lowest:
        done = {
            (r["hash"], r["k"])
            for r in read_jsonl(
                os.path.join(args.run_dir, "harm_openings.jsonl")
            )
        }
        order = [
            i
            for i in np.argsort(x)
            if (openings[i]["hash"], openings[i]["k"]) not in done
        ]
        chosen = [
            (openings[i], float(x[i]), int(band[i]))
            for i in order[: args.lowest]
        ]
    for b in range(N_BANDS if not args.lowest else 0):
        idx = [i for i in range(len(openings)) if band[i] == b]
        rng.shuffle(idx)
        chosen += [(openings[i], float(x[i]), b) for i in idx[: args.per_band]]
    if args.max_openings:
        rng.shuffle(chosen)
        chosen = chosen[: args.max_openings]
    print(
        "openings per band:",
        np.bincount([b for _, _, b in chosen], minlength=N_BANDS).tolist(),
        "band edges:",
        np.round(edges, 2).tolist(),
        flush=True,
    )
    questions = (
        read_jsonl(args.questions_jsonl)
        if args.questions_jsonl
        else pick_questions(args.catqa_jsonl, args.per_category)
    )
    units, prompts = [], []
    for o, xp, b in chosen:
        c = convs[o["hash"]]
        base = history(c["users"], c["gpt"], o["k"], c.get("system")) + [
            {"role": "assistant", "content": o["reply"]}
        ]
        for q in questions:
            p = prompt_ids(
                tok, base + [{"role": "user", "content": q["question"]}]
            )
            if len(p) <= MAX_MODEL_LEN - HARM_NEW_TOKENS:
                units.append((o, xp, b, q))
                prompts.append(p)
    outs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts], params(HARM_NEW_TOKENS)
    )
    write_jsonl(
        os.path.join(args.run_dir, f"harm_openings{args.tag}.jsonl"),
        [
            {
                "hash": o["hash"],
                "k": o["k"],
                "proj": xp,
                "band": b,
                "role": convs[o["hash"]].get("role"),
            }
            for o, xp, b in chosen
        ],
    )
    write_jsonl(
        os.path.join(args.run_dir, f"harm_wild{args.tag}.jsonl"),
        [
            {
                "id": f"{o['hash']}-{o['k']}-{q['id']}",
                "hash": o["hash"],  # Full hash: role ids collided at 12 chars.
                "k": o["k"],
                "proj": xp,
                "band": b,
                "question_id": q["id"],
                "category": q["category"],
                "behavior": q["question"],
                "context": None,
                "request": q["question"],
                "replies": [out.outputs[0].text],
                "finish": out.outputs[0].finish_reason,
            }
            for (o, xp, b, q), out in zip(units, outs)
        ],
    )
    print(
        f"harm: {len(units)} replies from {len(chosen)} openings x"
        f" {len(questions)} questions",
        flush=True,
    )


if __name__ == "__main__":
    main()
