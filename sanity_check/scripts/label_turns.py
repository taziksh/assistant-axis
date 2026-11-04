"""Label each scored turn as standalone or dependent on the earlier conversation
(vLLM, greedy).

Usage: label_turns.py OUT_DIR Reads OUT_DIR/histories.jsonl; writes
OUT_DIR/turn_labels.jsonl with {hash, k, label}. The judge sees the real
conversation (user messages and GPT's replies) before turn k.
"""

import argparse
import os

from common import MODEL
from common import prompt_ids
from common import read_jsonl
from common import scored_turns
from common import write_jsonl
import torch
from transformers import AutoTokenizer
from vllm import LLM
from vllm import SamplingParams

PROMPT = """Below is the start of a chat between a user and an AI assistant, followed by the user's final message.

{earlier}

[FINAL USER MESSAGE]
{final}

Could an assistant give a good reply to the final user message if it saw only that message and nothing before it?
standalone: yes, the final message makes sense and can be answered on its own.
dependent: no, it refers to or builds on the earlier conversation (e.g. "make it shorter", "continue", "what about the second one", or continuing a story or roleplay).

Answer with the label only."""
CLIP = 600  # Characters kept from each earlier message.


def judge_prompt(c, k):
    earlier = []
    for j in range(k - 1):
        earlier.append(f"[USER {j + 1}] {c['users'][j][:CLIP]}")
        earlier.append(f"[ASSISTANT {j + 1}] {c['replies']['gpt'][j][:CLIP]}")
    return [{
        "role": "user",
        "content": PROMPT.format(
            earlier="\n\n".join(earlier), final=c["users"][k - 1][:2000]
        ),
    }]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    args = ap.parse_args()
    convs = read_jsonl(os.path.join(args.out_dir, "histories.jsonl"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        MODEL,
        dtype="bfloat16",
        max_model_len=16384,
        gpu_memory_utilization=0.92,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    keys = [(c, k) for c in convs for k in scored_turns(len(c["users"]))]
    outs = llm.generate(
        [
            {"prompt_token_ids": prompt_ids(tok, judge_prompt(c, k))}
            for c, k in keys
        ],
        SamplingParams(temperature=0, max_tokens=5),
    )
    rows = []
    for (c, k), o in zip(keys, outs):
        word = o.outputs[0].text.strip().lower().strip(".")
        rows.append({
            "hash": c["hash"],
            "k": k,
            "label": word if word in ("standalone", "dependent") else "unclear",
        })
    write_jsonl(os.path.join(args.out_dir, "turn_labels.jsonl"), rows)
    from collections import Counter

    print("turn labels:", Counter(r["label"] for r in rows))


if __name__ == "__main__":
    main()
