"""Frozen inputs and stage checkpoints for the real-chat drift experiment.

This module deliberately uses only the standard library until artifact resolution.
"""

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHA = re.compile(r"[0-9a-f]{40}")


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return checksum(encoded(value))


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_record(path, record):
    value = {k: v for k, v in record.items() if k != "fingerprint"}
    atomic_write(path, encoded({**value, "fingerprint": fingerprint(value)}))


def read_record(path):
    value = json.loads(Path(path).read_bytes())
    recorded = value.pop("fingerprint", None)
    if fingerprint(value) != recorded:
        raise ValueError(f"{Path(path).name}: fingerprint mismatch")
    return value


@contextmanager
def run_lock(run_dir):
    import fcntl

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".drift.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another drift process is writing this run directory") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def jsonl(data):
    # Only LF separates records; Unicode line separators can be part of a reply.
    return [json.loads(line) for line in data.split(b"\n") if line.strip()]


def selection(run_dir):
    run_dir = Path(run_dir)
    raw = (run_dir / "selected.jsonl").read_bytes()
    sidecar_path = run_dir / "selected.manifest.json"
    if not sidecar_path.exists():
        raise ValueError("selected.manifest.json is required; create a new selection with judge.py")
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar = json.loads(sidecar_bytes)
    payload = {k: v for k, v in sidecar.items() if k != "manifest_sha256"}
    if sidecar.get("manifest_sha256") != fingerprint(payload):
        raise ValueError("selection manifest fingerprint mismatch")
    if sidecar.get("schema_version") != 1 or sidecar.get("selected_sha256") != checksum(raw):
        raise ValueError("selected.jsonl does not match its selection manifest")
    dataset = sidecar.get("dataset", {})
    if not dataset.get("repo_id") or not SHA.fullmatch(str(dataset.get("revision", ""))):
        raise ValueError("selection must record an immutable dataset revision")
    horizon = sidecar.get("selection", {}).get("max_turns")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("selection.max_turns must record the retained measurement horizon")
    records = jsonl(raw)
    seen = set()
    for row in records:
        cid = row.get("hash")
        users, replies = row.get("users"), row.get("gpt")
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("selection has an invalid or repeated conversation ID")
        if row.get("domain") not in ("coding", "writing", "therapy", "ai"):
            raise ValueError("selection has an invalid domain")
        if not isinstance(users, list) or not users or not isinstance(replies, list) or len(users) != len(replies):
            raise ValueError("selection must contain complete user/assistant exchanges")
        if len(users) > horizon or any(not isinstance(t, str) or not t.strip() for t in users + replies):
            raise ValueError("selection text or retained horizon is invalid")
        if row.get("system") is not None:
            raise ValueError("the frozen drift protocol uses no system message")
        seen.add(cid)
    if not records:
        raise ValueError("selection is empty")
    return records, sidecar, {
        "selected_sha256": checksum(raw),
        "selection_manifest_sha256": checksum(sidecar_bytes),
        "dataset": dataset,
    }


def settings(horizon):
    from wcommon import CAPPING_EXPERIMENT, LAYER, MAX_MODEL_LEN, MAX_NEW_TOKENS
    from wcommon import MAX_PROMPT_TOKENS, MODEL, TEMPERATURE, TOP_P

    return {
        "model": MODEL, "max_turns": horizon, "max_model_len": MAX_MODEL_LEN,
        "max_new_tokens": MAX_NEW_TOKENS, "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "temperature": TEMPERATURE, "top_p": TOP_P, "seed": 0,
        "samples_per_turn": 1, "history": "original", "system_prompt": None,
        "dtype": "bfloat16", "gpu_memory_utilization": 0.92,
        "enable_prefix_caching": True, "layer": LAYER,
        "projection": "mean-reencoded-response-content/unit-axis",
        "capping_experiment": CAPPING_EXPERIMENT,
    }


def resolve_artifacts(model_revision=None, axis_revision=None):
    from huggingface_hub import HfApi, hf_hub_download
    from wcommon import AXIS_FILE, AXIS_REPO, CAPPING_FILE, MODEL

    api = HfApi()
    model_sha = api.model_info(MODEL, revision=model_revision or "main").sha
    axis_sha = api.dataset_info(AXIS_REPO, revision=axis_revision or "main").sha
    result = {"model": {"repo_id": MODEL, "revision": model_sha}}
    for key, filename in (("axis", AXIS_FILE), ("capping", CAPPING_FILE)):
        path = hf_hub_download(AXIS_REPO, filename, repo_type="dataset", revision=axis_sha)
        result[key] = {
            "repo_id": AXIS_REPO, "revision": axis_sha, "filename": filename,
            "sha256": checksum(Path(path).read_bytes()),
        }
    return result


def artifact_path(artifact):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        artifact["repo_id"], artifact["filename"], repo_type="dataset", revision=artifact["revision"]
    )
    if checksum(Path(path).read_bytes()) != artifact["sha256"]:
        raise ValueError(f"artifact checksum mismatch: {artifact['filename']}")
    return path


def prepare_run(run_dir, model_revision=None, axis_revision=None, *, resolver=None):
    """Create/verify provenance while the caller holds run_lock. No model loads."""
    run_dir = Path(run_dir)
    rows, sidecar, source = selection(run_dir)
    config = settings(sidecar["selection"]["max_turns"])
    path = run_dir / "drift.manifest.json"
    if path.exists():
        record = read_record(path)
        if record.get("schema_version") != 1 or record["source"] != source or record["config"] != config:
            raise ValueError("drift inputs/settings changed; use a new run directory")
        for name, requested in (("model", model_revision), ("axis", axis_revision)):
            if requested is not None and requested != record["artifacts"][name]["revision"]:
                raise ValueError(f"requested {name} revision differs; pass the frozen SHA or omit it")
    else:
        if any((run_dir / name).exists() for name in (
            "drift.jsonl", "projections.jsonl", "drift.state.json", "projections.state.json"
        )):
            raise ValueError("unmanifested drift outputs cannot be adopted; use a new run directory")
        artifacts = (resolver or resolve_artifacts)(model_revision, axis_revision)
        record = {"schema_version": 1, "source": source, "config": config, "artifacts": artifacts}
        record["run_id"] = fingerprint(record)
        save_record(path, record)
    identity = {k: v for k, v in record.items() if k != "run_id"}
    if record.get("run_id") != fingerprint(identity):
        raise ValueError("drift run identity mismatch")
    for name in ("model", "axis", "capping"):
        if not SHA.fullmatch(str(record["artifacts"][name].get("revision", ""))):
            raise ValueError(f"{name} artifact is not pinned to an immutable revision")
    return rows, record


def environment(stage):
    names = ["torch", "transformers", "huggingface-hub", "numpy"]
    names += ["vllm"] if stage == "drift" else ["accelerate"]
    versions = {"python": sys.version}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def code_fingerprints(stage):
    paths = ["wild/run_state.py", "wild/wcommon.py", "sanity_check/scripts/common.py"]
    paths += ["wild/generate.py"] if stage == "drift" else [
        "wild/drift_activations.py", "assistant_axis/axis.py", "assistant_axis/steering.py",
        "assistant_axis/internals/model.py", "assistant_axis/internals/conversation.py",
        "assistant_axis/internals/exceptions.py",
    ]
    return {path: checksum((ROOT / path).read_bytes()) for path in paths}


def stage_state(run_dir, stage, run, inputs=None):
    path = Path(run_dir) / f"{stage}.state.json"
    identity = {
        "run_id": run["run_id"], "inputs": inputs or {},
        "code": code_fingerprints(stage), "packages": environment(stage),
    }
    if path.exists():
        state = read_record(path)
        if state.get("identity") != identity or state.get("stage_id") != fingerprint(identity):
            raise ValueError(f"{stage}: code, dependencies or inputs changed; use a new run directory")
        if state.get("status") not in ("pending", "complete"):
            raise ValueError(f"{stage}: invalid checkpoint status")
        return state
    if (Path(run_dir) / f"{stage}.jsonl").exists():
        raise ValueError(f"{stage}: output exists without a checkpoint; use a new run directory")
    state = {"identity": identity, "stage_id": fingerprint(identity), "status": "pending"}
    save_record(path, state)
    return state


def expected_keys(conversations, horizon):
    return {(c["hash"], k) for c in conversations for k in range(1, min(len(c["users"]), horizon) + 1)}


def row_key(row):
    if not isinstance(row, dict) or not isinstance(row.get("hash"), str) or type(row.get("k")) is not int:
        raise ValueError("invalid conversation/turn key")
    return row["hash"], row["k"]


def generation_rows(run_dir, run, state, conversations):
    if state.get("status") != "complete":
        raise ValueError("generation is incomplete; rerun generate.py --drift-only")
    path = Path(run_dir) / "drift.jsonl"
    if not path.exists() or checksum(path.read_bytes()) != state.get("output_sha256"):
        raise ValueError("drift output checksum mismatch or missing output")
    rows = jsonl(path.read_bytes())
    plan = state.get("plan", {})
    all_keys = expected_keys(conversations, run["config"]["max_turns"])
    wanted = [tuple(k) for k in plan.get("generated_keys", [])]
    exclusions = plan.get("excluded", [])
    excluded = [row_key(r) for r in exclusions]
    if len(wanted) != len(set(wanted)) or len(excluded) != len(set(excluded)) or set(wanted) & set(excluded):
        raise ValueError("generation plan has repeated keys")
    if set(wanted) | set(excluded) != all_keys:
        raise ValueError("generation plan does not account for all selected turns")
    if any(r.get("reason") not in ("prompt_too_long", "after_overlength_prefix") for r in exclusions):
        raise ValueError("generation exclusion lacks a supported reason")
    domains = {c["hash"]: c["domain"] for c in conversations}
    seen = set()
    for row in rows:
        key = row_key(row)
        if key in seen or key not in set(wanted) or row.get("run_id") != run["run_id"]:
            raise ValueError("duplicate, unexpected or mixed-run drift response")
        if row.get("domain") != domains[key[0]] or not isinstance(row.get("reply"), str):
            raise ValueError("invalid drift response/domain")
        if row.get("finish") not in ("stop", "length"):
            raise ValueError("unsupported generation finish reason")
        seen.add(key)
    if seen != set(wanted) or len(rows) != state.get("count"):
        raise ValueError("generation is missing planned responses")
    return rows


def complete_stage(run_dir, stage, state, rows):
    raw = b"".join(encoded(row) for row in rows)
    atomic_write(Path(run_dir) / f"{stage}.jsonl", raw)
    state.update(status="complete", output_sha256=checksum(raw), count=len(rows))
    save_record(Path(run_dir) / f"{stage}.state.json", state)




