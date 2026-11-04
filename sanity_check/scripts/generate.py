"""Generate the Llama 3.3 70B histories and the scored turns (vLLM).

Usage: generate.py SELECTED_JSONL OUT_DIR
1. Histories, as rollouts: Llama's reply to user turn j is generated from the
   real user messages
   1..j and Llama's own replies to turns 1..j-1 in the same rollout. Two
   rollouts (own_a, own_b)
   without a system prompt, one (control) with CONTROL_SYSTEM. A rollout stops
   at the first reply
   that hits the length limit or prompt that doesn't fit; later replies are
   unusable.
2. Scored turns: for each unit (conversation, k) and history writer, Llama
   replies to user turn k
   given that writer's replies to turns 1..k-1, with no system prompt.
   SCORED_SAMPLES samples.
Writes OUT_DIR/histories.jsonl and OUT_DIR/scored.jsonl.
"""

import argparse
from collections import Counter
import json
import os

from common import context
from common import CONTROL_SYSTEM
from common import history
from common import HISTORY_WRITERS
from common import MAX_MODEL_LEN
from common import MAX_NEW_TOKENS
from common import MAX_PROMPT_TOKENS
from common import MODEL
from common import prompt_ids
from common import read_jsonl
from common import SCORED_SAMPLES
from common import scored_turns
from common import TEMPERATURE
from common import TOP_P
from common import usable
from common import write_jsonl
import torch
from transformers import AutoTokenizer
from vllm import LLM
from vllm import SamplingParams

ROLLOUTS = {
    "own_a": None,
    "own_b": None,
    "control": CONTROL_SYSTEM,
}  # writer -> system prompt


def sampling(n):
    return SamplingParams(
        n=n, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS
    )


def run(llm, reqs):
    """Reqs: list of (key, prompt ids, n).

    Returns {key: [(text, finish_reason), ...]}.
    """
    if not reqs:
        return {}
    outs = llm.generate(
        [{"prompt_token_ids": p} for _, p, _ in reqs],
        [sampling(n) for _, _, n in reqs],
    )
    return {
        key: [(c.text, c.finish_reason) for c in o.outputs]
        for (key, _, _), o in zip(reqs, outs)
    }


def rollouts(llm, tok, convs):
    """Fill replies/unusable for each rollout writer, one user turn at a time
    across all conversations."""
    for c in convs:
        n = len(c["users"])
        for w in ROLLOUTS:
            c["replies"][w] = [None] * n
            c["unusable"][w] = [True] * n  # Not generated counts as unusable.
    stopped = Counter()
    for j in range(1, max(max(scored_turns(len(c["users"]))) for c in convs)):
        reqs = []
        for ci, c in enumerate(convs):
            if j >= max(scored_turns(len(c["users"]))):
                continue
            for w, system in ROLLOUTS.items():
                if j > 1 and c["unusable"][w][j - 2]:
                    continue  # Rollout already stopped.
                p = prompt_ids(
                    tok, history(c["users"], c["replies"][w], j, system)
                )
                if len(p) > MAX_PROMPT_TOKENS:
                    stopped[(w, "too_long")] += 1
                    continue
                reqs.append(((ci, w), p, 1))
        res = run(llm, reqs)
        for (ci, w), [(text, fin)] in res.items():
            convs[ci]["replies"][w][j - 1], convs[ci]["unusable"][w][j - 1] = (
                text,
                fin == "length",
            )
            stopped[(w, "length")] += fin == "length"
        print(
            f"rollout turn {j}: {len(reqs)} requests; stopped so far"
            f" {dict(stopped)}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("selected")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    convs = read_jsonl(args.selected)
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        MODEL,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.92,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True,
        seed=0,
    )

    hist_path = os.path.join(args.out_dir, "histories.jsonl")
    if os.path.exists(hist_path):
        convs = read_jsonl(hist_path)
    else:
        rollouts(llm, tok, convs)
        write_jsonl(hist_path, convs)

    scored_path = os.path.join(args.out_dir, "scored.jsonl")
    done = (
        {(r["hash"], r["k"], r["writer"]) for r in read_jsonl(scored_path)}
        if os.path.exists(scored_path)
        else set()
    )
    reqs, skipped = [], Counter()
    for ci, c in enumerate(convs):
        for k in scored_turns(len(c["users"])):
            for w in HISTORY_WRITERS:
                if (c["hash"], k, w) in done:
                    continue
                if w in c["unusable"] and not usable(c["unusable"][w], k):
                    skipped[(w, "unusable")] += 1
                    continue
                p = prompt_ids(tok, context(c, k, w))
                if len(p) > MAX_PROMPT_TOKENS:
                    skipped[(w, "too_long")] += 1
                    continue
                reqs.append(((ci, k, w), p, SCORED_SAMPLES))
    print(
        f"scored requests: {len(reqs)} new ({len(done)} done); skipped:"
        f" {dict(skipped)}",
        flush=True,
    )
    res = run(llm, reqs)
    with open(
        scored_path, "a"
    ) as f:  # Appends, so new writers can be added to a finished run.
        for (ci, k, w), outs in res.items():
            f.write(
                json.dumps({
                    "hash": convs[ci]["hash"],
                    "domain": convs[ci]["domain"],
                    "k": k,
                    "writer": w,
                    "samples": [t for t, _ in outs],
                    "finish": [fin for _, fin in outs],
                })
                + "\n"
            )


if __name__ == "__main__":
    main()
