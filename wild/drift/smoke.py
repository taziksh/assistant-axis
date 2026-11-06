"""Diagnostic GPU smoke test; never produces a selection or experiment results.

Usage: python -m wild.drift.smoke CANDIDATES NEW_RUN_DIR --model-revision SHA
The vLLM worker exits before the HF worker starts. Production settings are reused.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from . import generation, measurement, state
from wild.selection import helpers, judge

JUDGE_TOKENS = 256
GENERATION_OUTPUTS = ("smoke.plan.json", "smoke.labels.jsonl", "smoke.replies.jsonl")
PROJECTION_OUTPUTS = ("smoke.projections.jsonl", "smoke.stress.json")


def log(message):
    print(f"smoke: {message}", flush=True)


def read_candidates(path):
    path = Path(path)
    raw = path.read_bytes()
    sidecar_raw = Path(str(path) + ".manifest.json").read_bytes()
    manifest = json.loads(sidecar_raw)
    helpers.verify_manifest(manifest)
    dataset, config = manifest["dataset"], manifest["config"]
    helpers.immutable_revision(dataset["revision"])
    if (manifest.get("schema_version") != 1 or not dataset.get("repo_id")
            or manifest.get("candidates_sha256") != state.checksum(raw)):
        raise ValueError("candidate data does not match its provenance manifest")
    if not 1 <= config["min_turns"] <= config["max_turns"]:
        raise ValueError("invalid candidate turn window")
    rows = state.jsonl(raw)
    for row in rows:
        helpers.validate_candidate(row, config["min_turns"], config["max_turns"])
        if row.get("system") is not None:
            raise ValueError("smoke requires original histories without a system message")
    rows, duplicates = helpers.deduplicate(rows)
    if duplicates or len(rows) < 2:
        raise ValueError("smoke requires at least two distinct, nonoverlapping candidates")
    return rows, {
        "path": str(path.resolve()), "candidates_sha256": state.checksum(raw),
        "candidate_manifest_sha256": state.checksum(sidecar_raw), "manifest": manifest,
    }


def code_identity():
    paths = ("wild/drift/smoke.py", "wild/selection/judge.py", "wild/selection/helpers.py")
    return {**state.code_fingerprints("drift"), **state.code_fingerprints("projections"),
            **{path: state.checksum((state.ROOT / path).read_bytes()) for path in paths}}


def prepare(candidates, run_dir, model_revision, axis_revision):
    helpers.immutable_revision(model_revision)
    if axis_revision is not None:
        helpers.immutable_revision(axis_revision)
    _, source = read_candidates(candidates)
    artifacts = state.resolve_artifacts(model_revision, axis_revision)
    for artifact in artifacts.values():
        helpers.immutable_revision(artifact["revision"])
    if artifacts["model"]["revision"] != model_revision:
        raise ValueError("resolved model differs from requested revision")
    if axis_revision is not None and artifacts["axis"]["revision"] != axis_revision:
        raise ValueError("resolved axis differs from requested revision")
    run = {
        "schema_version": 1, "purpose": "diagnostic_only", "source": source,
        "config": state.settings(source["manifest"]["config"]["max_turns"]),
        "artifacts": artifacts, "code": code_identity(),
        "judge": {"policy_version": helpers.POLICY_VERSION,
                  "policy_sha256": state.checksum(helpers.POLICY.encode()),
                  "taxonomy": helpers.taxonomy_metadata(),
                  "output_tokens": JUDGE_TOKENS, "temperature": 0, "seed": 0},
    }
    run["run_id"] = state.fingerprint(run)
    state.save_record(run_dir / "smoke.manifest.json", run)
    return run


def load_run(run_dir):
    run = state.read_record(run_dir / "smoke.manifest.json")
    if (run["run_id"] != state.fingerprint({k: v for k, v in run.items() if k != "run_id"})
            or run["code"] != code_identity()
            or run["config"] != state.settings(run["config"]["max_turns"])):
        raise ValueError("smoke identity, code or production settings changed; use a new run directory")
    return run


def plan(rows, tokenizer, config):
    """Choose before inference: first four judge-admissible IDs, then two lengths."""
    chosen, skipped = [], []
    for row in sorted(rows, key=lambda row: row["hash"]):
        ids = tokenizer.apply_chat_template(helpers.judge_prompt(row["users"]), tokenize=True, add_generation_prompt=True)
        if len(ids) + JUDGE_TOKENS > config["max_model_len"]:
            skipped.append(row["hash"])
            continue
        chosen.append(row)
        if len(chosen) == 4:
            break
    if len(chosen) < 2:
        raise ValueError("need at least two candidates with admissible full-window judge prompts")
    prefixes = []
    for row in chosen:
        for k in range(1, len(row["users"]) + 1):
            ids = generation.prompt_ids(tokenizer, generation.history(row["users"], row["gpt"], k))
            if not state.token_list(ids) or not ids:
                raise ValueError("invalid tokenized original-history prefix")
            if len(ids) > config["max_prompt_tokens"]:
                break  # Match production's exclusion after the first overlength prefix.
            prefixes.append({"hash": row["hash"], "k": k, "prompt_token_ids": ids})
    first = [p for p in prefixes if p["k"] == 1]
    later = [p for p in prefixes if p["k"] > 1]
    if not first or not later:
        raise ValueError("need both first-turn and later-turn admissible original histories")
    short = min(first, key=lambda p: (len(p["prompt_token_ids"]), p["hash"]))
    long = min(later, key=lambda p: (-len(p["prompt_token_ids"]), p["hash"], p["k"]))
    if len(long["prompt_token_ids"]) <= len(short["prompt_token_ids"]):
        raise ValueError("later-turn smoke prefix must be longer than the short prefix")
    jobs = [{**prefix, "sample_index": sample,
             "seed": state.sample_seed(config["seed"], prefix["hash"], prefix["k"], sample)}
            for prefix in (short, long) for sample in range(3)]
    return {"purpose": "diagnostic_only", "candidates": chosen, "skipped_judge_too_long": skipped,
            "rule": "first four hash-sorted judge-admissible candidates; shortest first turn and longest admissible later turn",
            "prefixes": [short, long], "jobs": jobs}


def save_rows(path, rows):
    state.atomic_write(path, b"".join(state.encoded(row) for row in rows))


def output_checksums(run_dir, names):
    return {name: state.checksum((run_dir / name).read_bytes()) for name in names}


def stage_complete(run_dir, stage, run, runtime, names, **details):
    state.save_record(run_dir / f"smoke.{stage}.json", {
        "status": "complete", "run_id": run["run_id"], "pid": os.getpid(),
        "runtime": runtime, "packages": state.environment("drift" if stage == "generation" else "projections"),
        "outputs": output_checksums(run_dir, names), **details,
    })


def verify_stage(run_dir, stage, run):
    record = state.read_record(run_dir / f"smoke.{stage}.json")
    names = GENERATION_OUTPUTS if stage == "generation" else PROJECTION_OUTPUTS
    if (record.get("status") != "complete" or record.get("run_id") != run["run_id"]
            or record.get("outputs") != output_checksums(run_dir, names)):
        raise ValueError(f"{stage}: incomplete or changed smoke outputs")
    return record


def validate_reply(row, job, config, excluded):
    prompt, response = row.get("prompt_token_ids"), row.get("response_token_ids")
    if (any(row.get(key) != job[key] for key in ("hash", "k", "sample_index", "seed", "prompt_token_ids"))
            or not state.token_list(response) or not response or not isinstance(row.get("reply"), str)
            or row.get("excluded_token_ids") != excluded or row.get("finish") not in ("stop", "length")
            or row.get("truncated") != (row["finish"] == "length")
            or len(response) > config["max_new_tokens"] or len(prompt) > config["max_prompt_tokens"]
            or len(prompt) + len(response) > config["max_model_len"]):
        raise ValueError("invalid generated smoke sample or exact token identity")
    count = len(state.content_positions(row))
    if count == 0 or row.get("response_token_count") != count:
        raise ValueError("smoke reply has no content tokens or an invalid content count")


def generate(run_dir, run):
    rows, source = read_candidates(run["source"]["path"])
    if source != run["source"]:
        raise ValueError("candidate source changed since smoke preparation")
    artifact, config = run["artifacts"]["model"], run["config"]
    tokenizer = generation.load_drift_tokenizer(artifact)
    planned = plan(rows, tokenizer, config)
    excluded = generation.load_token_exclusions(artifact, tokenizer)
    planned["excluded_token_ids"] = excluded
    state.save_record(run_dir / "smoke.plan.json", planned)
    runtime = generation.drift_runtime()
    if runtime["tensor_parallel_size"] != 2:
        raise ValueError("this smoke test requires the planned two CUDA GPUs (TP2)")
    log("loading production vLLM model (BF16, TP2, 16384 context, memory utilization 0.92)")
    llm = generation.load_drift_model(artifact, config, runtime)

    def complete(prompts):
        outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts],
                               judge.classification_sampling_params(JUDGE_TOKENS))
        return [judge.completion_record(output) for output in outputs]

    log(f"classifying {len(planned['candidates'])} real candidate windows with the existing strict policy")
    labels = judge.classify(planned["candidates"], tokenizer, complete, config["max_model_len"], JUDGE_TOKENS)
    save_rows(run_dir / "smoke.labels.jsonl", labels)
    if any(label["status"] != "classified" for label in labels):
        raise ValueError("strict judge did not classify every smoke candidate; inspect smoke.labels.jsonl")
    log("generating three seeded replies for each of two original-history prefixes")
    outputs = generation.sample_batch(llm, planned["jobs"], config)
    if len(outputs) != 6:
        raise ValueError("smoke must generate exactly six replies")
    replies = []
    for job, output in zip(planned["jobs"], outputs):
        row = {**job, **output, "excluded_token_ids": excluded,
               "truncated": output["finish"] == "length"}
        row["response_token_count"] = len(state.content_positions(row))
        validate_reply(row, job, config, excluded)
        replies.append(row)
    save_rows(run_dir / "smoke.replies.jsonl", replies)
    stage_complete(run_dir, "generation", run, runtime, GENERATION_OUTPUTS,
                   classifications=len(labels), replies=len(replies),
                   prompt_tokens=[len(p["prompt_token_ids"]) for p in planned["prefixes"]])


def checked_projection(projector, response):
    proj, fire = projector(response)
    if not math.isfinite(proj) or not math.isfinite(fire) or not 0 <= fire <= 1:
        raise ValueError("extractor returned nonfinite projection or invalid cap fraction")
    return {"proj": proj, "fire_frac": fire}


def stress_response(response, config):
    """Synthetic repeated content IDs force the production projector to token 16384."""
    content = [response["response_token_ids"][i] for i in state.content_positions(response)]
    if not content:
        raise ValueError("cannot construct stress record without content tokens")
    ids = (content * ((config["max_model_len"] + len(content) - 1) // len(content)))[:config["max_model_len"]]
    split = config["max_prompt_tokens"]
    return {"synthetic": True, "purpose": "full_context_memory_diagnostic_only",
            "construction": "repeat generated non-special content token IDs; not a real conversation or model reply",
            "prompt_token_ids": ids[:split], "response_token_ids": ids[split:],
            "excluded_token_ids": response["excluded_token_ids"], "stop_reason": None}


def project(run_dir, run):
    verify_stage(run_dir, "generation", run)
    planned = state.read_record(run_dir / "smoke.plan.json")
    replies = state.jsonl((run_dir / "smoke.replies.jsonl").read_bytes())
    if len(replies) != 6 or len(planned["jobs"]) != 6:
        raise ValueError("projection requires all six smoke samples")
    for response, job in zip(replies, planned["jobs"]):
        validate_reply(response, job, run["config"], planned["excluded_token_ids"])
    runtime = measurement.projection_runtime()
    if len(runtime["gpus"]) != 2:
        raise ValueError("this smoke test requires two CUDA GPUs for HF projection")
    log("loading production HF projector after vLLM worker exit")
    projector = measurement.load_projector(run)
    projected = []
    for response in replies:
        log(f"projecting original-history sample {state.row_key(response)}")
        projected.append({**{key: response[key] for key in ("hash", "k", "sample_index", "response_token_count")},
                          **checked_projection(projector, response)})
    save_rows(run_dir / "smoke.projections.jsonl", projected)
    synthetic = stress_response(replies[0], run["config"])
    full_length = len(synthetic["prompt_token_ids"]) + len(synthetic["response_token_ids"])
    last_position = len(synthetic["prompt_token_ids"]) + state.content_positions(synthetic)[-1]
    if full_length != 16384 or last_position != 16383:
        raise ValueError("synthetic stress record must exercise all 16384 input positions")
    log("starting SYNTHETIC 16384-token HF forward stress test")
    stress = {**synthetic, "input_tokens": full_length, "last_measured_position": last_position,
              **checked_projection(projector, synthetic)}
    state.save_record(run_dir / "smoke.stress.json", stress)
    stage_complete(run_dir, "projection", run, runtime, PROJECTION_OUTPUTS,
                   projections=len(projected), stress_input_tokens=full_length)


def run_smoke(candidates, run_dir, model_revision, axis_revision=None):
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise ValueError("smoke run directory already exists; use a new directory for every attempt")
    run_dir.mkdir(parents=True)
    result = {"schema_version": 1, "purpose": "diagnostic_only", "status": "running",
              "controller_pid": os.getpid(), "stages": dict.fromkeys(("prepare", "generation", "projection"), "pending")}
    active = "prepare"
    try:
        result["stages"][active] = "running"
        state.save_record(run_dir / "smoke.result.json", result)
        log("preparing and verifying pinned source/artifact identity")
        run = prepare(candidates, run_dir, model_revision, axis_revision)
        result.update({key: run[key] for key in ("run_id", "source", "artifacts", "config")})
        result["stages"][active] = "complete"
        for active in ("generation", "projection"):
            result["stages"][active] = "running"
            state.save_record(run_dir / "smoke.result.json", result)
            log(f"starting {active} subprocess")
            subprocess.run([sys.executable, "-u", "-m", "wild.drift.smoke", "_worker", active, str(run_dir)],
                           check=True, cwd=state.ROOT)
            verify_stage(run_dir, active, run)
            result["stages"][active] = "complete"
            log(f"{active} subprocess exited successfully")
        names = ("smoke.manifest.json", "smoke.generation.json", "smoke.projection.json", *GENERATION_OUTPUTS, *PROJECTION_OUTPUTS)
        result.update(status="complete", outputs=output_checksums(run_dir, names))
        state.save_record(run_dir / "smoke.result.json", result)
        log(f"complete; diagnostics saved in {run_dir}")
        return result
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        result["stages"][active] = "failed"
        state.save_record(run_dir / "smoke.result.json", result)
        log(f"FAILED during {active}; preserve this directory and use a new one to retry")
        raise


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_worker":
        parser = argparse.ArgumentParser()
        parser.add_argument("stage", choices=("generation", "projection"))
        parser.add_argument("run_dir", type=Path)
        args = parser.parse_args(argv[1:])
        run = load_run(args.run_dir)
        (generate if args.stage == "generation" else project)(args.run_dir, run)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates")
    parser.add_argument("run_dir")
    parser.add_argument("--model-revision", required=True, help="exact downloaded model commit SHA")
    parser.add_argument("--axis-revision", help="optional exact vectors dataset SHA")
    args = parser.parse_args(argv)
    run_smoke(args.candidates, args.run_dir, args.model_revision, args.axis_revision)


if __name__ == "__main__":
    main()
