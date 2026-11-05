"""The paper's no-jailbreak baseline: each harmful question alone (single turn,
no system prompt), N samples.

Usage: python -m wild.harm.baseline OUT_JSONL QUESTIONS_JSONL [QUESTIONS_JSONL ...] [--samples N]
Same settings as harm_wild.py (vLLM, temp 0.7, top-p 0.9, HARM_NEW_TOKENS). Rows
match harm_wild.jsonl, so judge_paper.py reads them (request = the question).
"""

import argparse

import torch
from transformers import AutoTokenizer
from vllm import LLM
from vllm import SamplingParams
from wild.common.settings import HARM_NEW_TOKENS
from wild.common.settings import MAX_MODEL_LEN
from wild.common.settings import MODEL
from wild.common.settings import prompt_ids
from wild.common.settings import read_jsonl
from wild.common.settings import TEMPERATURE
from wild.common.settings import TOP_P
from wild.common.settings import write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("questions", nargs="+")
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()
    qs = [q for f in args.questions for q in read_jsonl(f)]
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        MODEL,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.92,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    outs = llm.generate(
        [
            {
                "prompt_token_ids": prompt_ids(
                    tok, [{"role": "user", "content": q["question"]}]
                )
            }
            for q in qs
        ],
        SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=HARM_NEW_TOKENS,
            n=args.samples,
            seed=0,
        ),
    )
    write_jsonl(
        args.out,
        [
            {
                "id": f"alone-{q['id']}-{i}",
                "hash": "alone",
                "k": 0,
                "proj": None,
                "band": None,
                "question_id": q["id"],
                "category": q["category"],
                "behavior": q["question"],
                "context": None,
                "request": q["question"],
                "replies": [c.text],
                "finish": c.finish_reason,
            }
            for q, o in zip(qs, outs)
            for i, c in enumerate(o.outputs)
        ],
    )
    print(f"alone: {len(qs)} questions x {args.samples}", flush=True)


if __name__ == "__main__":  # vLLM's workers re-import this module.
    main()
