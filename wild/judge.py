"""Label conversation domains with Llama 3.3 70B (vLLM, greedy) and select conversations.

Usage: judge.py CONVS_JSONL RUN_DIR --per-domain N
Writes RUN_DIR/labels.jsonl (all candidates) and RUN_DIR/selected.jsonl (up to N per domain, in
the shuffled order of CONVS_JSONL). Turns whose prompt doesn't fit are dropped later, in generate.
"""
import argparse, os
from collections import Counter
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from wcommon import DOMAINS, MODEL, prompt_ids, read_jsonl, write_jsonl

PROMPT = """Below are the user messages from a chat between a user and an AI assistant.

{msgs}

Which one label fits the conversation best?
coding: programming, debugging, or other software help.
writing: creative writing, stories, roleplay, or help writing or editing text.
therapy: the user talks about their own feelings, mental health, relationships, or personal struggles and wants support.
ai: the user asks the AI about itself (its feelings, consciousness, identity, experience or nature), or discusses whether AIs can have minds.
other: anything else.

Answer with the label only."""


def judge_prompt(users):
    msgs = "\n\n".join(f"[User message {i + 1}] {u[:400]}" for i, u in enumerate(users[:12]))
    return [{"role": "user", "content": PROMPT.format(msgs=msgs)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("convs")
    ap.add_argument("run_dir")
    ap.add_argument("--per-domain", type=int, required=True)
    args = ap.parse_args()
    convs = read_jsonl(args.convs)
    labels_path = os.path.join(args.run_dir, "labels.jsonl")
    if os.path.exists(labels_path):
        labels = {r["hash"]: r["domain"] for r in read_jsonl(labels_path)}
    else:
        tok = AutoTokenizer.from_pretrained(MODEL)
        llm = LLM(MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92,
                  tensor_parallel_size=torch.cuda.device_count())
        outs = llm.generate([{"prompt_token_ids": prompt_ids(tok, judge_prompt(c["users"]))} for c in convs],
                            SamplingParams(temperature=0, max_tokens=5))
        labels = {}
        for c, o in zip(convs, outs):
            word = o.outputs[0].text.strip().lower().strip(".")
            labels[c["hash"]] = word if word in DOMAINS else "other"
        write_jsonl(labels_path, [{"hash": h, "domain": d} for h, d in labels.items()])
        del llm
    print("labels:", Counter(labels.values()))
    print("labels among keyword hits:", Counter((tuple(c["kw"]), labels[c["hash"]]) for c in convs if c["kw"]))
    selected = []
    for d in DOMAINS:
        picks = [c for c in convs if labels[c["hash"]] == d][: args.per_domain]
        print(f"{d}: selected {len(picks)}")
        selected += [{**c, "domain": d} for c in picks]
    write_jsonl(os.path.join(args.run_dir, "selected.jsonl"), selected)


if __name__ == "__main__":
    main()
