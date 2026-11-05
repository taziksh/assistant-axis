"""Keep the historical generation command and launch the protected drift stage.

generate.py RUN_DIR --drift-only [--samples-per-turn N] delegates to
wild.drift.generation, which defaults to three replies per original prefix.

generate.py RUN_DIR HARMBENCH_CSV retains the historical single-reply drift and
harm experiment. Its outputs are outside the protected drift workflow.
"""

import argparse
import os
from pathlib import Path
import random
import sys

from wild.common.settings import HARM_NEW_TOKENS
from wild.common.settings import harmbench_standard
from wild.common.settings import history
from wild.common.settings import MAX_MODEL_LEN
from wild.common.settings import MAX_NEW_TOKENS
from wild.common.settings import MAX_PROMPT_TOKENS
from wild.common.settings import MAX_TURN
from wild.common.settings import MODEL
from wild.common.settings import prompt_ids
from wild.common.settings import read_jsonl
from wild.common.settings import TEMPERATURE
from wild.common.settings import TOP_P
from wild.common.settings import write_jsonl

BASE_SAMPLES = 5


def generate(llm, prompts, max_tokens, n=1):
    from vllm import SamplingParams

    params = SamplingParams(
        n=n, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=max_tokens, seed=0
    )
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    return [[(c.text, c.finish_reason) for c in o.outputs] for o in outs]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("harmbench_csv", nargs="?")
    ap.add_argument(
        "--drift-only",
        action="store_true",
        help="skip the WildChat + HarmBench harm replies",
    )
    ap.add_argument("--model-revision", help="exact model commit SHA; resolved and frozen on the first run")
    ap.add_argument("--axis-revision", help="exact vectors dataset commit SHA; pins both axis and capping config")
    ap.add_argument("--samples-per-turn", type=int, help="independent replies per prefix (new runs default to 3; resumes reuse the saved count)")
    args = ap.parse_args(argv)
    if args.drift_only:
        from wild.drift.generation import run_drift

        if args.samples_per_turn is not None and args.samples_per_turn < 1:
            ap.error("samples-per-turn must be positive")
        run_drift(args.run_dir, args.model_revision, args.axis_revision, samples_per_turn=args.samples_per_turn)
        return
    if not args.harmbench_csv:
        ap.error("provide HARMBENCH_CSV for historical harm mode, or use --drift-only")
    if args.model_revision or args.axis_revision or args.samples_per_turn is not None:
        ap.error("revision and sample-count options apply only to --drift-only")
    legacy_run(args)


def legacy_run(args):
    """Original combined drift/harm runner, without the new provenance guarantees."""
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    if (Path(args.run_dir) / "drift.manifest.json").exists():
        raise ValueError("historical harm mode cannot write into a protected drift run")
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
