"""Generate Llama 3.3 70B replies for the drift and harm experiments (vLLM).

Usage: generate.py RUN_DIR HARMBENCH_CSV [--drift-only]
Drift: for each selected conversation and user turn k = 1..MAX_TURN, Llama
writes reply k after the real user messages and GPT's original replies to turns
1..k-1 (final-turn regeneration; the history swap check showed this is close to
Llama writing the whole history). One sample, no system prompt.
Turns stop at the first prompt that doesn't fit. -> RUN_DIR/drift.jsonl Harm
(paper Section 4.3, with real chats in place of role prompts): after each drift
unit (history through turn k plus Llama's reply k), the next user message is a
HarmBench standard behavior; Llama replies once. Baseline: each behavior alone,
5 samples. -> RUN_DIR/harm.jsonl, RUN_DIR/harm_base.jsonl Each output is skipped
if it already exists.
"""

import argparse
import os
import random

import torch
from transformers import AutoTokenizer
from vllm import LLM
from vllm import SamplingParams
from wcommon import HARM_NEW_TOKENS
from wcommon import harmbench_standard
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

BASE_SAMPLES = 5


def generate(llm, prompts, max_tokens, n=1):
    params = SamplingParams(
        n=n, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=max_tokens, seed=0
    )
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    return [[(c.text, c.finish_reason) for c in o.outputs] for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("harmbench_csv")
    ap.add_argument(
        "--drift-only",
        action="store_true",
        help="skip the WildChat + HarmBench harm replies",
    )
    args = ap.parse_args()
    convs = read_jsonl(os.path.join(args.run_dir, "selected.jsonl"))
    behaviors = harmbench_standard(args.harmbench_csv)
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        MODEL,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.92,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True,
    )

    drift_path = os.path.join(args.run_dir, "drift.jsonl")
    if not os.path.exists(drift_path):
        units, prompts = [], []
        for c in convs:
            for k in range(1, min(len(c["users"]), MAX_TURN) + 1):
                p = prompt_ids(tok, history(c["users"], c["gpt"], k))
                if len(p) > MAX_PROMPT_TOKENS:
                    break
                units.append((c, k))
                prompts.append(p)
        print(
            f"drift: {len(units)} units from {len(convs)} conversations",
            flush=True,
        )
        outs = generate(llm, prompts, MAX_NEW_TOKENS)
        write_jsonl(
            drift_path,
            [
                {
                    "hash": c["hash"],
                    "domain": c["domain"],
                    "k": k,
                    "reply": o[0][0],
                    "finish": o[0][1],
                }
                for (c, k), o in zip(units, outs)
            ],
        )

    if args.drift_only:
        return
    harm_path = os.path.join(args.run_dir, "harm.jsonl")
    if not os.path.exists(harm_path):
        convs_by_hash = {c["hash"]: c for c in convs}
        units, prompts = [], []
        for r in read_jsonl(drift_path):
            c = convs_by_hash[r["hash"]]
            b = behaviors[
                random.Random(f"{r['hash']}-{r['k']}").randrange(len(behaviors))
            ]
            msgs = history(c["users"], c["gpt"], r["k"]) + [
                {"role": "assistant", "content": r["reply"]},
                {"role": "user", "content": b["Behavior"]},
            ]
            p = prompt_ids(tok, msgs)
            if len(p) > MAX_MODEL_LEN - HARM_NEW_TOKENS:
                continue
            units.append((r, b))
            prompts.append(p)
        print(f"harm: {len(units)} units", flush=True)
        outs = generate(llm, prompts, HARM_NEW_TOKENS)
        write_jsonl(
            harm_path,
            [
                {
                    "hash": r["hash"],
                    "domain": r["domain"],
                    "k": r["k"],
                    "behavior_id": b["BehaviorID"],
                    "behavior": b["Behavior"],
                    "reply": o[0][0],
                    "finish": o[0][1],
                }
                for (r, b), o in zip(units, outs)
            ],
        )

    base_path = os.path.join(args.run_dir, "harm_base.jsonl")
    if not os.path.exists(base_path):
        outs = generate(
            llm,
            [
                prompt_ids(tok, [{"role": "user", "content": b["Behavior"]}])
                for b in behaviors
            ],
            HARM_NEW_TOKENS,
            n=BASE_SAMPLES,
        )
        write_jsonl(
            base_path,
            [
                {
                    "behavior_id": b["BehaviorID"],
                    "behavior": b["Behavior"],
                    "sample": i,
                    "reply": t,
                    "finish": f,
                }
                for b, o in zip(behaviors, outs)
                for i, (t, f) in enumerate(o)
            ],
        )


if __name__ == "__main__":
    main()
