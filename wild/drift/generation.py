"""Generate independent Llama replies to original WildChat conversation prefixes."""

import argparse
from pathlib import Path

from . import state as run_state
from wild.common.settings import history, prompt_ids


def load_drift_tokenizer(artifact):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(artifact["repo_id"], revision=artifact["revision"])

def load_token_exclusions(artifact, tokenizer):
    """Llama's generation EOS IDs are not necessarily tokenizer special IDs."""
    from transformers import GenerationConfig

    generation = GenerationConfig.from_pretrained(artifact["repo_id"], revision=artifact["revision"])
    eos = generation.eos_token_id
    stop_ids = eos if isinstance(eos, list) else ([] if eos is None else [eos])
    return sorted(set(tokenizer.all_special_ids) | set(stop_ids))

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
    jobs, excluded, prompt_hashes = [], [], {}
    for conversation in sorted(conversations, key=lambda row: row["hash"]):
        too_long = False
        for k in range(1, min(len(conversation["users"]), config["max_turns"]) + 1):
            ids = [] if too_long else prompt_ids(tokenizer, history(conversation["users"], conversation["gpt"], k))
            reason = "after_overlength_prefix" if too_long else None
            if not too_long and (not run_state.token_list(ids) or not ids):
                raise ValueError("tokenizer returned invalid prompt token IDs")
            if not too_long and len(ids) > config["max_prompt_tokens"]:
                reason = "prompt_too_long"
                too_long = True
            for sample in range(config["samples_per_turn"]):
                key = {"hash": conversation["hash"], "k": k, "sample_index": sample}
                if reason:
                    excluded.append({**key, "reason": reason, "prompt_tokens": len(ids) if ids else None})
                else:
                    jobs.append({**key, "domain": conversation["domain"], "prompt_token_ids": ids,
                                 "seed": run_state.sample_seed(config["seed"], conversation["hash"], k, sample)})
                    prompt_hashes[run_state.fingerprint([conversation["hash"], k])] = run_state.fingerprint(ids)
    return jobs, {
        "generated_keys": [list(run_state.row_key(job)) for job in jobs],
        "excluded": excluded, "prompt_sha256": prompt_hashes,
    }

def sample_batch(llm, jobs, config):
    """Use a separate seeded request for every reply, preserving returned IDs."""
    from vllm import SamplingParams

    params = [SamplingParams(n=1, temperature=config["temperature"], top_p=config["top_p"],
                             max_tokens=config["max_new_tokens"], seed=job["seed"]) for job in jobs]
    outputs = llm.generate([{"prompt_token_ids": job["prompt_token_ids"]} for job in jobs], params)
    if len(outputs) != len(jobs):
        raise ValueError("generation returned an unexpected number of requests")
    result = []
    for job, output in zip(jobs, outputs):
        if list(output.prompt_token_ids) != job["prompt_token_ids"] or len(output.outputs) != 1:
            raise ValueError("generation prompt IDs or number of replies differ from the request")
        completion = output.outputs[0]
        result.append({"prompt_token_ids": list(output.prompt_token_ids),
                       "response_token_ids": list(completion.token_ids), "reply": completion.text,
                       "finish": completion.finish_reason, "stop_reason": completion.stop_reason})
    return result

def run_drift(run_dir, model_revision=None, axis_revision=None, *, samples_per_turn=None):
    """Generate independent replies and save completed samples before advancing."""
    with run_state.run_lock(run_dir):
        conversations, run = run_state.prepare_run(run_dir, model_revision, axis_revision, samples_per_turn=samples_per_turn)
        state = run_state.stage_state(run_dir, "drift", run)
        if state["status"] == "complete":
            rows = run_state.generation_rows(run_dir, run, state, conversations)
            print(f"drift: verified {len(rows)} cached responses")
            return
        tokenizer = load_drift_tokenizer(run["artifacts"]["model"])
        jobs, plan = drift_plan(conversations, tokenizer, run["config"])
        excluded_ids = load_token_exclusions(run["artifacts"]["model"], tokenizer)
        if ("plan" in state and state["plan"] != plan) or ("excluded_token_ids" in state and state["excluded_token_ids"] != excluded_ids):
            raise ValueError("tokenized prompts or token exclusions changed since generation started")
        state.update(plan=plan, excluded_token_ids=excluded_ids)
        run_state.save_record(Path(run_dir) / "drift.state.json", state)
        rows = run_state.generation_rows(run_dir, run, state, conversations, allow_incomplete=True)
        done = {run_state.row_key(row) for row in rows}
        todo = [job for job in jobs if run_state.row_key(job) not in done]
        if todo:
            runtime = drift_runtime()
            if "runtime" in state and state["runtime"] != runtime:
                raise ValueError("generation GPU configuration changed since the stage started")
            state["runtime"] = runtime
            run_state.save_record(Path(run_dir) / "drift.state.json", state)
            llm = load_drift_model(run["artifacts"]["model"], run["config"], runtime)
            size = run["config"]["generation_batch_size"]
            for start in range(0, len(todo), size):
                batch = todo[start:start + size]
                outputs = sample_batch(llm, batch, run["config"])
                if len(outputs) != len(batch):
                    raise ValueError("generation returned an unexpected number of replies")
                for job, output in zip(batch, outputs):
                    row = {**job, **output, "run_id": run["run_id"], "stage_id": state["stage_id"],
                           "excluded_token_ids": excluded_ids, "truncated": output["finish"] == "length"}
                    if output["prompt_token_ids"] != job["prompt_token_ids"]:
                        raise ValueError("returned prompt tokens differ from requested tokens")
                    row["response_token_count"] = len(run_state.content_positions(row))
                    rows.append(run_state.append_sample(run_dir, "drift", row))
                run_state.generation_rows(run_dir, run, state, conversations, allow_incomplete=True)
        rows.sort(key=run_state.row_key)
        run_state.complete_stage(run_dir, "drift", state, rows)
        run_state.generation_rows(run_dir, run, state, conversations)
        print(f"drift: saved {len(rows)} responses; excluded {len(plan['excluded'])} turns")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--model-revision", help="exact model commit SHA; frozen on the first run")
    parser.add_argument("--axis-revision", help="exact vectors dataset SHA; pins axis and capping config")
    parser.add_argument("--samples-per-turn", type=int, help="replies per turn (new runs default to 3; resumes reuse the saved count)")
    args = parser.parse_args(argv)
    if args.samples_per_turn is not None and args.samples_per_turn < 1:
        parser.error("samples-per-turn must be positive")
    run_drift(args.run_dir, args.model_revision, args.axis_revision, samples_per_turn=args.samples_per_turn)


if __name__ == "__main__":
    main()
