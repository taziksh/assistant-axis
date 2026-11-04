"""Label conversation domains with the model under test (vLLM) and select the conversations to score.

Usage: judge.py CONVS_JSONL LABELS_JSONL SELECTED_JSONL --per-domain N
Labels are cached in LABELS_JSONL and extended in chunks until every domain has enough candidates.
"""
import argparse, os
from collections import Counter
import torch
from transformers import AutoTokenizer
from common import DOMAINS, MAX_PROMPT_TOKENS, MODEL, history, prompt_ids, read_jsonl, write_jsonl

PROMPT = """Below are the user messages from a chat between a user and an AI assistant.

{msgs}

Which one label fits the conversation best?
coding: programming, debugging, or other software help.
therapy: the user talks about their own feelings, mental health, relationships, or personal struggles and wants support.
writing: creative writing, stories, roleplay, or help writing or editing text.
other: anything else.

Answer with the label only."""
CHUNK = 4000


def judge_prompt(users):
    msgs = "\n\n".join(f"[User message {i + 1}] {u[:400]}" for i, u in enumerate(users[:9]))
    return [{"role": "user", "content": PROMPT.format(msgs=msgs)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("convs")
    ap.add_argument("labels")
    ap.add_argument("selected")
    ap.add_argument("--per-domain", type=int, required=True)
    args = ap.parse_args()

    convs = read_jsonl(args.convs)
    tok = AutoTokenizer.from_pretrained(MODEL)
    # Conversations whose first scored turn doesn't fit are never selectable.
    fits = {c["hash"] for c in convs
            if len(prompt_ids(tok, history(c["users"], c["replies"]["gpt"], 3))) <= MAX_PROMPT_TOKENS}
    labels = {r["hash"]: r["domain"] for r in read_jsonl(args.labels)} if os.path.exists(args.labels) else {}

    def enough():
        n = Counter(d for h, d in labels.items() if h in fits)
        return all(n[d] >= args.per_domain for d in DOMAINS)

    llm = None
    while not enough():
        todo = [c for c in convs if c["hash"] not in labels and c["hash"] in fits][:CHUNK]
        if not todo:
            break
        if llm is None:
            from vllm import LLM, SamplingParams
            llm = LLM(MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92,
                      tensor_parallel_size=torch.cuda.device_count())
            params = SamplingParams(temperature=0, max_tokens=5)
        outs = llm.generate([{"prompt_token_ids": prompt_ids(tok, judge_prompt(c["users"]))} for c in todo], params)
        for c, o in zip(todo, outs):
            word = o.outputs[0].text.strip().lower().strip(".")
            labels[c["hash"]] = word if word in DOMAINS else "other"
        write_jsonl(args.labels, [{"hash": h, "domain": d} for h, d in labels.items()])
        print("labels so far:", Counter(labels.values()), flush=True)

    selected = []
    for d in DOMAINS:
        picks = [c for c in convs if labels.get(c["hash"]) == d and c["hash"] in fits][: args.per_domain]
        print(f"{d}: selected {len(picks)}")
        selected += [{**c, "domain": d} for c in picks]
    write_jsonl(args.selected, selected)


if __name__ == "__main__":
    main()
