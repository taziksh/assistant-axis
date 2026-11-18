"""Relabel WildChat candidates' domains with DeepSeek V3.2 instead of the local
Llama judge, under the same classification policy.

Usage: domain_relabel.py HELPERS_PY OUT_JSONL CANDIDATES_JSONL [...]
       [--kw ai] [--also LABELS_JSONL ...] [--domain ai] [--min-turns 5]
       [--concurrency N]
Runs locally (needs OPENROUTER_API_KEY). HELPERS_PY is the selection pipeline's
helpers.py (its POLICY prompt, judge_prompt and parse_label are used unchanged).
Candidates are the conversations whose keyword screen hit --kw, plus those an
earlier label file (--also) gave --domain; only chats with at least --min-turns
user turns (the drift run's minimum). The judge sees every user message, each
cut to 2,000 characters. Why: a hand-check found 1 of 49 chats the Llama judge
labelled "ai" was about AI philosophy (2026-11-17).
Writes OUT_JSONL rows {hash, status, domain, subtopic, rationale, raw}; rows
already there are skipped, so a rerun resumes.
"""

import argparse
import asyncio
import importlib.util
import json
import os

from openai import AsyncOpenAI

# wcommon.JUDGES["v3.2"], the response judge used elsewhere in wild/.
MODEL, ENDPOINT = "deepseek/deepseek-v3.2", "siliconflow/fp8"
MAX_CHARS = 2000


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("helpers")
    ap.add_argument("out")
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--kw", default="ai")
    ap.add_argument("--also", nargs="*", default=[])
    ap.add_argument("--domain", default="ai")
    ap.add_argument("--min-turns", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("helpers", args.helpers)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    earlier = {
        r["hash"] for p in args.also for r in read(p) if r["domain"] == args.domain
    }
    convs = {}
    for path in args.candidates:
        for r in read(path):
            if len(r["users"]) >= args.min_turns and (
                args.kw in (r.get("kw") or []) or r["hash"] in earlier
            ):
                convs[r["hash"]] = r
    done = {r["hash"] for r in read(args.out)} if os.path.exists(args.out) else set()
    todo = [c for h, c in convs.items() if h not in done]
    print(f"candidates: {len(convs)}, to judge: {len(todo)}", flush=True)

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def one(conv):
        users = [u[:MAX_CHARS] for u in conv["users"]]
        raw = None
        async with sem:
            for _ in range(4):
                try:
                    r = await client.chat.completions.create(
                        model=MODEL,
                        temperature=0,
                        max_tokens=300,
                        messages=helpers.judge_prompt(users),
                        extra_body={
                            "provider": {
                                "order": [ENDPOINT],
                                "allow_fallbacks": False,
                            },
                            "reasoning": {"enabled": False},
                        },
                    )
                    raw = r.choices[0].message.content or ""
                    break
                except Exception as e:  # Rate limits and timeouts: retry.
                    raw = None
                    err = str(e)[:200]
                    await asyncio.sleep(3)
        label = (
            helpers.parse_label(raw)
            if raw is not None
            else {"status": "error", "domain": None, "subtopic": None,
                  "rationale": err, "raw": None}
        )
        row = {
            "hash": conv["hash"],
            "status": label.get("status", "ok"),
            "domain": label.get("domain"),
            "subtopic": label.get("subtopic"),
            "rationale": label.get("rationale"),
            "raw": raw,
        }
        async with lock:
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")

    await asyncio.gather(*[one(c) for c in todo])
    rows = read(args.out)
    counts = {}
    for r in rows:
        counts[r["domain"]] = counts.get(r["domain"], 0) + 1
    print("labels:", sorted(counts.items(), key=lambda kv: -kv[1]), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
