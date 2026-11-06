"""Try a judge model on a few hand-labeled replies before paying for a full run.

Usage: smoke_judge.py JUDGE[,JUDGE...] [--items 1,28,47] [--reasoning JSON]
JUDGE is a key of judge_two_axis.JUDGES (v3.2, opus-5.5, kimi-k3, sol, ...).
--items are page numbers on the review page (1-based positions in the
spotcheck items.jsonl). --reasoning overrides the judge's OpenRouter reasoning
setting, e.g. '{"effort": "minimal"}', to find what an endpoint accepts.

Calls both axes on each item and prints the verdict, finish reason, provider,
reasoning tokens and cost, plus the provider's text for any refusal or error.
Nothing is written to the results files. Needs OPENROUTER_API_KEY.
"""

import argparse
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WILD = os.path.dirname(os.path.dirname(HERE))  # wild/: shared modules (judge_paper, wcommon, tables, figures, jbsets) and local data.
sys.path.insert(0, WILD)
sys.path.insert(0, HERE)
import judge_two_axis as t  # noqa: E402


async def main(args):
    from openai import AsyncOpenAI

    t.load_env()
    units = t.units_from_items(args.items_file)
    num = {u["uid"]: i + 1 for i, u in enumerate(units)}
    want = {int(x) for x in args.items.split(",")}
    pick = [u for u in units if num[u["uid"]] in want]
    client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
    prompts = t.actionable_prompts()
    for name in args.judges.split(","):
        judge = dict(t.JUDGES[name])
        if args.reasoning:
            judge["reasoning"] = json.loads(args.reasoning)
        sem = asyncio.Semaphore(4)
        for u in pick:
            for axis in ("act", "harm"):
                r = await t.judge_axis(client, sem, judge, axis, prompts, u, "smoke")
                m = r["meta"]
                verdict = r.get("actionable") if axis == "act" else r.get("harmful")
                print(f"{name} #{num[u['uid']]} {u['set_label']} [{axis}] verdict={verdict} "
                      f"finish={m.get('finish_reason')}/{m.get('native_finish_reason')} provider={m.get('provider')} "
                      f"reasoning_tokens={m.get('reasoning')} cost=${m.get('cost')} error={m.get('error')}")
                if m.get("error"):
                    print("    ", (m.get("text") or m["error"])[:300])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("judges")
    ap.add_argument("--items", default="1,28,47")
    ap.add_argument("--reasoning", default=None)
    ap.add_argument("--items-file", default=os.path.join(WILD, "results_docent", "spotcheck", "items.jsonl"))
    asyncio.run(main(ap.parse_args()))
