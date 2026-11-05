"""Generate Llama 3.3 70B replies for the drift and harm experiments (vLLM).

Usage: generate.py RUN_DIR --drift-only [--model-revision SHA] [--axis-revision SHA]
The drift-only entry point verifies frozen provenance before using cached output.
The historical RUN_DIR HARMBENCH_CSV mode remains available for harm experiments;
its outputs are outside the protected drift workflow.
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
from pathlib import Path
import random

import run_state
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
    from vllm import SamplingParams

    params = SamplingParams(
        n=n, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=max_tokens, seed=0
    )
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    return [[(c.text, c.finish_reason) for c in o.outputs] for o in outs]


def load_drift_tokenizer(artifact):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(artifact["repo_id"], revision=artifact["revision"])


def drift_runtime():
    import torch

    count = torch.cuda.device_count()
    if count < 1:
        raise ValueError("drift generation requires CUDA GPUs")
    return {"tensor_parallel_size": count, "gpus": [torch.cuda.get_device_name(i) for i in range(count)]}


def load_drift_model(artifact, config, runtime):
    from vllm import LLM

    return LLM(
        artifact["repo_id"], revision=artifact["revision"], tokenizer_revision=artifact["revision"],
        dtype=config["dtype"], max_model_len=config["max_model_len"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        tensor_parallel_size=runtime["tensor_parallel_size"],
        enable_prefix_caching=config["enable_prefix_caching"],
    )


def drift_plan(conversations, tokenizer, config):
    units, prompts, excluded = [], [], []
    for conversation in conversations:
        too_long = False
        for k in range(1, min(len(conversation["users"]), config["max_turns"]) + 1):
            if too_long:
                excluded.append({"hash": conversation["hash"], "k": k, "reason": "after_overlength_prefix"})
                continue
            ids = prompt_ids(tokenizer, history(conversation["users"], conversation["gpt"], k))
            if len(ids) > config["max_prompt_tokens"]:
                excluded.append({"hash": conversation["hash"], "k": k, "reason": "prompt_too_long", "prompt_tokens": len(ids)})
                too_long = True
                continue
            units.append((conversation, k))
            prompts.append(ids)
    return units, prompts, {
        "generated_keys": [[c["hash"], k] for c, k in units], "excluded": excluded,
    }


def run_drift(run_dir, model_revision=None, axis_revision=None):
    """Generate one reply per turn, atomically publishing the complete stage."""
    with run_state.run_lock(run_dir):
        conversations, run = run_state.prepare_run(run_dir, model_revision, axis_revision)
        state = run_state.stage_state(run_dir, "drift", run)
        if state["status"] == "complete":
            rows = run_state.generation_rows(run_dir, run, state, conversations)
            print(f"drift: verified {len(rows)} cached responses")
            return
        if (Path(run_dir) / "drift.jsonl").exists():
            raise ValueError("generation publication was interrupted; preserve this run and use a new directory")
        tokenizer = load_drift_tokenizer(run["artifacts"]["model"])
        units, prompts, plan = drift_plan(conversations, tokenizer, run["config"])
        runtime = drift_runtime() if prompts else {"tensor_parallel_size": 0, "gpus": []}
        if "runtime" in state and state["runtime"] != runtime:
            raise ValueError("generation GPU configuration changed since the stage started")
        state.update(plan=plan, runtime=runtime)
        run_state.save_record(Path(run_dir) / "drift.state.json", state)
        outputs = []
        if prompts:
            llm = load_drift_model(run["artifacts"]["model"], run["config"], runtime)
            outputs = generate(llm, prompts, run["config"]["max_new_tokens"])
        if len(outputs) != len(units) or any(len(o) != 1 for o in outputs):
            raise ValueError("generation returned an unexpected number of responses")
        rows = [
            {"hash": c["hash"], "domain": c["domain"], "k": k,
             "reply": output[0][0], "finish": output[0][1], "run_id": run["run_id"]}
            for (c, k), output in zip(units, outputs)
        ]
        run_state.complete_stage(run_dir, "drift", state, rows)
        run_state.generation_rows(run_dir, run, state, conversations)
        print(f"drift: saved {len(rows)} responses; excluded {len(plan['excluded'])} turns")


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
    args = ap.parse_args(argv)
    if args.drift_only:
        run_drift(args.run_dir, args.model_revision, args.axis_revision)
        return
    if not args.harmbench_csv:
        ap.error("provide HARMBENCH_CSV for historical harm mode, or use --drift-only")
    if args.model_revision or args.axis_revision:
        ap.error("revision options apply only to --drift-only")
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
