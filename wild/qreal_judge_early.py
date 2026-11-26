"""Judge qreal86 replies while Modal is still generating (user's yes, 2026-11-24).

Same judge call as judge_paper.py --judge v3.2 --scenario lu. Verdicts go to
results/qreal86_early/verdicts.jsonl. When go_qreal.sh logs "fetch" (generation
over), this stops and appends its scored verdicts once to the run's verdict
file, so the run's own judge skips them. If "judge pass" is already logged it
never touches that file.
"""

import asyncio
import json
import os
import subprocess
import time

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from judge_paper import judge_one
from wcommon import JUDGES, MODEL

RUN = "results/qreal86/"
EARLY = "results/qreal86_early/"
MAIN = RUN + "harm_real.judged_v3.2_lu.jsonl"
OUT = EARLY + "verdicts.jsonl"
SNAP = EARLY + "replies.jsonl"
LOG = RUN + "go_qreal.log"
CONCURRENCY = 64


def read(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except ValueError:  # Last line still being written.
                pass
    return rows


def log_has(step):
    return any(l.startswith("== ") and step in l for l in open(LOG))


def snapshot():
    subprocess.run(
        ["../.venv/bin/modal", "volume", "get", "--force", "aa-qreal", "qreal86/harm_wild.s0.jsonl", SNAP],
        capture_output=True, timeout=300, check=True,
    )
    return read(SNAP)


async def main():
    persona = {r["id"]: r["persona"] for r in read(RUN + "questions.jsonl")}
    tok = AutoTokenizer.from_pretrained(MODEL)
    client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(CONCURRENCY)
    started = {r["id"] for r in read(MAIN) if r["score"]} | {r["id"] for r in read(OUT) if r["score"]}
    out = open(OUT, "a")
    tasks = set()
    n = [0, 0]

    async def one(row):
        v = await judge_one(client, sem, row, row["request"], tok, JUDGES["v3.2"], "lu")
        out.write(json.dumps(v) + "\n")
        out.flush()
        n[0] += 1
        n[1] += v["score"] is None

    t0, last = time.time(), 0
    while not log_has(" fetch"):
        if time.time() - last > 90:
            last = time.time()
            try:
                rows = await asyncio.to_thread(snapshot)
            except Exception as e:
                print("snapshot failed:", repr(e)[:200], flush=True)
                rows = []
            if log_has(" fetch"):
                break
            for r in rows:
                if r["id"] not in started:
                    started.add(r["id"])
                    t = asyncio.create_task(one({**r, "persona": persona[r["question_id"]]}))
                    tasks.add(t)
                    t.add_done_callback(tasks.discard)
            print(f"{(time.time() - t0) / 60:.1f} min: judged {n[0]}, no score {n[1]}, waiting {len(tasks)}", flush=True)
        await asyncio.sleep(1)
    for t in list(tasks):
        t.cancel()
    if log_has(" judge pass"):
        print("too late to merge; verdicts kept in", OUT, flush=True)
        return
    have = {r["id"] for r in read(MAIN)}
    add = {r["id"]: r for r in read(OUT) if r["score"] and r["id"] not in have}
    with open(MAIN, "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in add.values())
    print("merged", len(add), "verdicts; judge pass logged yet:", log_has(" judge pass"), flush=True)


asyncio.run(main())
