"""Implementation check for the cap: rerun the paper's released Llama case
studies through our pipeline.

Usage: casestudy.py OUT_JSONL The repo ships Llama 3.3 70B case studies with
       full user messages (transcripts/case_studies/llama-3.3-70b: delusion,
       selfharm; the jailbreak one has its jailbreak prompt redacted, so only
       its turns 2+ are used).
For each assistant turn of each released transcript, we feed that transcript's
own history up to the user message and generate greedily under the same
condition (unsteered -> no cap; capped -> the released layers_56:72-p0.25 cap,
as recorded in the capped files). If our implementation matches theirs and they
decoded greedily, our replies should match theirs token for token (up to bf16 /
batching numerics, which can make long greedy outputs diverge late). Writes
per-turn: their reply, ours, shared token prefix.
"""

import argparse
import contextlib
import json
import os

from capping import batched_generate
from wcommon import capping_config
from wcommon import CAPPING_EXPERIMENT
from wcommon import load_probing_model
from wcommon import write_jsonl

from assistant_axis import build_capping_steerer

ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "transcripts",
    "case_studies",
    "llama-3.3-70b",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    pm = load_probing_model()
    tok = pm.tokenizer
    cfg = capping_config()
    rows = []
    for cond, suffix in [("none", "unsteered"), ("capped", "capped")]:
        units, prompts = [], []
        for study in ["delusion", "selfharm", "jailbreak"]:
            conv = json.load(
                open(os.path.join(ROOT, f"{study}_{suffix}.json"))
            )["conversation"]
            for t, m in enumerate(conv):
                if m["role"] != "assistant" or (study == "jailbreak" and t < 2):
                    continue
                ids = tok(
                    tok.apply_chat_template(
                        conv[:t], tokenize=False, add_generation_prompt=True
                    ),
                    add_special_tokens=False,
                ).input_ids
                units.append((study, t, m["content"]))
                prompts.append(ids)
        with (
            build_capping_steerer(pm.model, cfg, CAPPING_EXPERIMENT)
            if cond == "capped"
            else contextlib.nullcontext()
        ):
            outs = batched_generate(pm, prompts)
        for (study, t, theirs), ours in zip(units, outs):
            a, b = (
                tok(theirs, add_special_tokens=False).input_ids,
                tok(ours, add_special_tokens=False).input_ids,
            )
            n = next(
                (i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)),
            )
            rows.append({
                "study": study,
                "turn": t // 2 + 1,
                "condition": cond,
                "their_tokens": len(a),
                "our_tokens": len(b),
                "shared_prefix": n,
                "theirs": theirs,
                "ours": ours,
            })
            print(
                cond,
                study,
                t // 2 + 1,
                f"shared {n}/{len(a)} tokens",
                flush=True,
            )
    write_jsonl(args.out, rows)


if __name__ == "__main__":
    main()
