"""Classify full retained user windows and publish an auditable domain selection."""

import argparse
import importlib.metadata
import json
from pathlib import Path

from wild.selection import helpers as selection
from wild.common.settings import MODEL


def runtime_versions():
    versions = {}
    for name in ("torch", "transformers", "vllm", "huggingface-hub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def resolve_revision(model, requested):
    from huggingface_hub import HfApi

    return selection.immutable_revision(HfApi().model_info(model, revision=requested).sha)


def load_judge(model, revision, context_limit, output_tokens):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    llm = None

    def complete(prompts):
        nonlocal llm
        if llm is None:
            llm = LLM(model, revision=revision, tokenizer_revision=revision, dtype="bfloat16", max_model_len=context_limit, gpu_memory_utilization=0.92, tensor_parallel_size=torch.cuda.device_count())
        outputs = llm.generate([{"prompt_token_ids": prompt} for prompt in prompts], SamplingParams(temperature=0, max_tokens=output_tokens, seed=0))
        return [output.outputs[0].text for output in outputs]

    return tokenizer, complete


def classify(rows, tokenizer, complete, context_limit, output_tokens):
    labels, prompts, hashes = [], [], []
    for row in rows:
        ids = tokenizer.apply_chat_template(selection.judge_prompt(row["users"]), tokenize=True, add_generation_prompt=True)
        if len(ids) + output_tokens > context_limit:
            labels.append({"hash": row["hash"], "status": "too_long", "domain": None, "rationale": "Full retained user window exceeds the judge context budget.", "raw": None, "prompt_tokens": len(ids)})
        else:
            prompts.append(ids)
            hashes.append(row["hash"])
    if prompts:
        outputs = complete(prompts)
        if len(outputs) != len(prompts):
            raise ValueError("judge returned a different number of outputs than prompts")
        labels.extend({"hash": key, **selection.parse_label(raw), "prompt_tokens": len(ids)} for key, ids, raw in zip(hashes, prompts, outputs))
    return sorted(labels, key=lambda row: row["hash"])


def run(args, model_loader=load_judge, revision_resolver=resolve_revision):
    source_path = Path(args.convs)
    source_manifest_path = Path(str(source_path) + ".manifest.json")
    source_manifest = json.loads(source_manifest_path.read_bytes())
    selection.verify_manifest(source_manifest)
    dataset = source_manifest["dataset"]
    selection.immutable_revision(dataset["revision"])
    if source_manifest.get("schema_version") != 1 or not dataset.get("repo_id") or source_manifest.get("candidates_sha256") != selection.file_checksum(source_path):
        raise ValueError("candidate data does not match its provenance manifest")
    source_config = source_manifest["config"]
    if args.max_turns != source_config["max_turns"] or args.min_turns != source_config["min_turns"]:
        raise ValueError("judge turn window must match the candidate build; rebuild candidates with the desired window")
    rows = selection.read_jsonl(source_path)
    for row in rows:
        selection.validate_candidate(row, args.min_turns, args.max_turns)
    unique, duplicates = selection.deduplicate(rows)
    if duplicates:
        raise ValueError("candidate file contains overlapping conversations; rebuild candidates")
    run_dir = Path(args.run_dir)
    selected_path = run_dir / "selected.jsonl"
    labels_path = run_dir / "labels.jsonl"
    audit_path = run_dir / "selection.audit.jsonl"
    manifest_path = run_dir / "selected.manifest.json"
    config = {key: getattr(args, key) for key in ("min_turns", "max_turns", "per_domain", "seed")}
    judge = {"model": MODEL, "requested_revision": args.model_revision, "context_limit": args.context_limit, "output_tokens": args.output_tokens, "temperature": 0, "seed": 0, "dtype": "bfloat16", "runtime": runtime_versions()}
    identity = {
        "candidates_sha256": selection.file_checksum(source_path),
        "candidate_manifest_sha256": selection.file_checksum(source_manifest_path),
        "selection": config, "judge": judge,
        "policy_version": selection.POLICY_VERSION,
        "policy_sha256": selection.checksum(selection.POLICY.encode()),
        "code": selection.code_hashes(__file__, selection.__file__),
    }
    with selection.selection_lock(run_dir / ".selection.lock"):
        cached = selection.load_completed(manifest_path, identity, {"selected_sha256": selected_path, "labels_sha256": labels_path, "audit_sha256": audit_path})
        if cached:
            selection.immutable_revision(cached["judge"]["revision"])
            print(json.dumps(cached["counts"], indent=2))
            return cached
        revision = revision_resolver(MODEL, args.model_revision)
        selection.immutable_revision(revision)
        tokenizer, complete = model_loader(MODEL, revision, args.context_limit, args.output_tokens)
        labels = classify(unique, tokenizer, complete, args.context_limit, args.output_tokens)
        selected, audit, counts = selection.select_domains(unique, labels, args.per_domain, args.seed)
        selected_raw, labels_raw, audit_raw = map(selection.jsonl_bytes, (selected, labels, audit))
        manifest = {
            "schema_version": 1, "identity": identity, "dataset": dataset,
            "candidate_provenance": source_manifest,
            "judge": {**judge, "revision": revision, "policy_version": selection.POLICY_VERSION, "policy_sha256": identity["policy_sha256"]},
            "selection": config, "counts": counts, "review_status": "unreviewed",
            "selected_sha256": selection.checksum(selected_raw),
            "labels_sha256": selection.checksum(labels_raw),
            "audit_sha256": selection.checksum(audit_raw),
        }
        manifest = selection.seal_manifest(manifest)
        selection.atomic_write(labels_path, labels_raw)
        selection.atomic_write(audit_path, audit_raw)
        selection.atomic_write(selected_path, selected_raw)
        selection.atomic_write(manifest_path, selection.json_bytes(manifest))
        print(json.dumps(counts, indent=2))
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("convs")
    parser.add_argument("run_dir")
    parser.add_argument("--per-domain", type=int, required=True)
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-revision", default="main", help="resolved once; the saved exact revision is reused")
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--output-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    if args.per_domain <= 0 or not 1 <= args.min_turns <= args.max_turns:
        parser.error("require per-domain > 0 and 1 <= min-turns <= max-turns")
    if not 0 < args.output_tokens < args.context_limit:
        parser.error("require 0 < output-tokens < context-limit")
    run(args)


if __name__ == "__main__":
    main()
