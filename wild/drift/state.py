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
ROOT = HERE.parent.parent
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


def settings(horizon, samples_per_turn=3):
    from wild.wcommon import CAPPING_EXPERIMENT, LAYER, MAX_MODEL_LEN, MAX_NEW_TOKENS
    from wild.wcommon import MAX_PROMPT_TOKENS, MODEL, TEMPERATURE, TOP_P

    if type(samples_per_turn) is not int or samples_per_turn < 1:
        raise ValueError("samples_per_turn must be a positive integer")
    return {
        "model": MODEL, "max_turns": horizon, "max_model_len": MAX_MODEL_LEN,
        "max_new_tokens": MAX_NEW_TOKENS, "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "temperature": TEMPERATURE, "top_p": TOP_P, "seed": 0,
        "samples_per_turn": samples_per_turn, "generation_batch_size": 16,
        "history": "original", "system_prompt": None,
        "dtype": "bfloat16", "gpu_memory_utilization": 0.92,
        "enable_prefix_caching": True, "layer": LAYER,
        "projection": "mean-generated-content-token-activations/unit-axis",
        "capping_experiment": CAPPING_EXPERIMENT,
    }


def resolve_artifacts(model_revision=None, axis_revision=None):
    from huggingface_hub import HfApi, hf_hub_download
    from wild.wcommon import AXIS_FILE, AXIS_REPO, CAPPING_FILE, MODEL

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


def prepare_run(run_dir, model_revision=None, axis_revision=None, *, samples_per_turn=None, resolver=None):
    """Create/verify provenance while the caller holds run_lock. No model loads."""
    run_dir = Path(run_dir)
    rows, sidecar, source = selection(run_dir)
    path = run_dir / "drift.manifest.json"
    if path.exists():
        record = read_record(path)
        if record.get("schema_version") != 2:
            raise ValueError("this run predates sample/token records; use a new run directory")
        count = record["config"]["samples_per_turn"] if samples_per_turn is None else samples_per_turn
        config = settings(sidecar["selection"]["max_turns"], count)
        if record["source"] != source or record["config"] != config:
            raise ValueError("drift inputs/settings changed; use a new run directory")
        for name, requested in (("model", model_revision), ("axis", axis_revision)):
            if requested is not None and requested != record["artifacts"][name]["revision"]:
                raise ValueError(f"requested {name} revision differs; pass the frozen SHA or omit it")
    else:
        config = settings(sidecar["selection"]["max_turns"], 3 if samples_per_turn is None else samples_per_turn)
        if any((run_dir / name).exists() for name in (
            "drift.jsonl", "projections.jsonl", "drift.state.json", "projections.state.json"
        )):
            raise ValueError("unmanifested drift outputs cannot be adopted; use a new run directory")
        artifacts = (resolver or resolve_artifacts)(model_revision, axis_revision)
        record = {"schema_version": 2, "source": source, "config": config, "artifacts": artifacts}
        record["run_id"] = fingerprint(record)
        save_record(path, record)
    validate_run(record, source)
    return rows, record


def validate_run(record, source):
    if record.get("schema_version") != 2 or record.get("source") != source:
        raise ValueError("run schema or frozen selection differs")
    identity = {k: v for k, v in record.items() if k != "run_id"}
    if record.get("run_id") != fingerprint(identity):
        raise ValueError("drift run identity mismatch")
    for name in ("model", "axis", "capping"):
        if not SHA.fullmatch(str(record["artifacts"][name].get("revision", ""))):
            raise ValueError(f"{name} artifact is not pinned to an immutable revision")
    config = record["config"]
    if type(config.get("samples_per_turn")) is not int or config["samples_per_turn"] < 1:
        raise ValueError("invalid saved sample count")


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
    paths = ["wild/drift/state.py", "wild/wcommon.py", "sanity_check/scripts/common.py"]
    paths += ["wild/drift/generation.py"] if stage == "drift" else [
        "wild/drift/measurement.py", "assistant_axis/axis.py", "assistant_axis/steering.py",
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


def expected_keys(conversations, horizon, samples_per_turn=3):
    return {(c["hash"], k, sample) for c in conversations
            for k in range(1, min(len(c["users"]), horizon) + 1)
            for sample in range(samples_per_turn)}


def row_key(row):
    if (not isinstance(row, dict) or not isinstance(row.get("hash"), str)
            or type(row.get("k")) is not int or row["k"] < 1
            or type(row.get("sample_index")) is not int or row["sample_index"] < 0):
        raise ValueError("invalid conversation/turn/sample key")
    return row["hash"], row["k"], row["sample_index"]


def sample_seed(base_seed, cid, turn, sample_index):
    return int(fingerprint([base_seed, cid, turn, sample_index])[:8], 16) & 0x7fffffff


def content_positions(response):
    excluded = set(response["excluded_token_ids"])
    if type(response.get("stop_reason")) is int:
        excluded.add(response["stop_reason"])
    return [i for i, token in enumerate(response["response_token_ids"]) if token not in excluded]


def checked_rows(run_dir, stage, state):
    path = Path(run_dir) / f"{stage}.jsonl"
    if not path.exists():
        if state["status"] == "complete":
            raise ValueError(f"completed {stage} output is missing")
        return []
    raw = path.read_bytes()
    if state["status"] == "complete" and checksum(raw) != state.get("output_sha256"):
        raise ValueError(f"{stage} output checksum mismatch")
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"{stage}.jsonl has an interrupted final write; preserve it and use a new run directory")
    rows = jsonl(raw)
    for row in rows:
        payload = {k: v for k, v in row.items() if k != "record_sha256"}
        if fingerprint(payload) != row.get("record_sha256"):
            raise ValueError(f"{stage} record checksum mismatch")
    return rows


def generation_rows(run_dir, run, state, conversations, *, allow_incomplete=False):
    if state.get("status") != "complete" and not allow_incomplete:
        raise ValueError("generation is incomplete; rerun generate.py --drift-only")
    rows = checked_rows(run_dir, "drift", state)
    plan = state.get("plan", {})
    config = run["config"]
    all_keys = expected_keys(conversations, config["max_turns"], config["samples_per_turn"])
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
    excluded_ids = state.get("excluded_token_ids")
    if not token_list(excluded_ids) or excluded_ids != sorted(set(excluded_ids)):
        raise ValueError("invalid recorded special/stop token IDs")
    seen = set()
    for row in rows:
        key = row_key(row)
        if key in seen or key not in set(wanted) or row.get("run_id") != run["run_id"] or row.get("stage_id") != state["stage_id"]:
            raise ValueError("duplicate, unexpected or mixed-run drift response")
        if row.get("domain") != domains[key[0]] or not isinstance(row.get("reply"), str):
            raise ValueError("invalid drift response/domain")
        if row.get("finish") not in ("stop", "length"):
            raise ValueError("unsupported generation finish reason")
        if type(row.get("seed")) is not int or row["seed"] != sample_seed(config["seed"], *key):
            raise ValueError("saved sample seed differs from the frozen experiment")
        prompt, response = row.get("prompt_token_ids"), row.get("response_token_ids")
        if not token_list(prompt) or not prompt or not token_list(response):
            raise ValueError("invalid saved prompt/response token IDs")
        if len(prompt) > config["max_prompt_tokens"] or len(response) > config["max_new_tokens"] or len(prompt) + len(response) > config["max_model_len"]:
            raise ValueError("saved sample exceeds the token budget")
        if row.get("excluded_token_ids") != excluded_ids:
            raise ValueError("saved sample token exclusions differ")
        if fingerprint(prompt) != plan.get("prompt_sha256", {}).get(fingerprint(list(key[:2]))):
            raise ValueError("saved prompt tokens differ from the frozen plan")
        if type(row.get("truncated")) is not bool or row["truncated"] != (row["finish"] == "length"):
            raise ValueError("saved truncation flag disagrees with finish reason")
        stop = row.get("stop_reason")
        if stop is not None and not (type(stop) is int and stop >= 0) and not isinstance(stop, str):
            raise ValueError("invalid saved stop reason")
        if type(row.get("response_token_count")) is not int or row["response_token_count"] != len(content_positions(row)):
            raise ValueError("saved content-token count differs from token IDs")
        seen.add(key)
    if state["status"] == "complete" and (seen != set(wanted) or len(rows) != state.get("count")):
        raise ValueError("generation is missing planned responses")
    return rows


def token_list(value):
    return isinstance(value, list) and all(type(token) is int and token >= 0 for token in value)


def complete_stage(run_dir, stage, state, rows):
    raw = b"".join(encoded(row) for row in rows)
    atomic_write(Path(run_dir) / f"{stage}.jsonl", raw)
    state.update(status="complete", output_sha256=checksum(raw), count=len(rows))
    save_record(Path(run_dir) / f"{stage}.state.json", state)


def projection_rows(run_dir, run, state, generated):
    rows = checked_rows(run_dir, "projections", state)
    source = {row_key(row): row for row in generated}
    seen = set()
    for row in rows:
        key = row_key(row)
        if key not in source or key in seen or row.get("run_id") != run["run_id"] or row.get("stage_id") != state["stage_id"]:
            raise ValueError("duplicate, unexpected or mixed-run projection")
        proj, fire = row.get("proj"), row.get("fire_frac")
        count = len(content_positions(source[key]))
        if type(row.get("response_token_count")) is not int or row["response_token_count"] != count:
            raise ValueError("projection content-token count differs from generated tokens")
        if type(row.get("truncated")) is not bool or row["truncated"] != source[key]["truncated"]:
            raise ValueError("projection truncation flag differs from generation")
        if count == 0:
            if proj is not None or fire is not None or row.get("reason") != "no_content_tokens":
                raise ValueError("empty content requires a documented null projection")
        elif type(proj) not in (int, float) or not math.isfinite(proj) or type(fire) not in (int, float) or not math.isfinite(fire) or not 0 <= fire <= 1:
            raise ValueError("invalid projection or cap firing fraction")
        seen.add(key)
    if state["status"] == "complete" and (seen != set(source) or len(rows) != state.get("count")):
        raise ValueError("completed projections do not cover the generated responses")
    return rows


def append_sample(run_dir, stage, row):
    row = {**row, "record_sha256": fingerprint(row)}
    with (Path(run_dir) / f"{stage}.jsonl").open("ab") as stream:
        stream.write(encoded(row))
        stream.flush()
        os.fsync(stream.fileno())
    return row


def append_projection(run_dir, row):
    return append_sample(run_dir, "projections", row)


def load_completed_run(run_dir):
    """Read a completed run on CPU; caller holds run_lock during analysis.

    Saved identities and links are checked, without requiring the reader's
    installed packages or source files to match the GPU execution environment.
    """
    run_dir = Path(run_dir)
    conversations, sidecar, source = selection(run_dir)
    run = read_record(run_dir / "drift.manifest.json")
    validate_run(run, source)
    if run["config"]["max_turns"] != sidecar["selection"]["max_turns"]:
        raise ValueError("measurement and selection horizons differ")

    def completed_state(stage, inputs):
        state = read_record(run_dir / f"{stage}.state.json")
        identity = state.get("identity", {})
        if (identity.get("run_id") != run["run_id"] or identity.get("inputs") != inputs
                or state.get("stage_id") != fingerprint(identity)):
            raise ValueError(f"{stage} stage does not match the saved experiment")
        if state.get("status") != "complete":
            raise ValueError(f"{stage} is incomplete; finish it before analysis")
        return state

    drift_state = completed_state("drift", {})
    generated = generation_rows(run_dir, run, drift_state, conversations)
    state = completed_state("projections", {
        "drift_sha256": drift_state["output_sha256"], "drift_stage_id": drift_state["stage_id"],
    })
    projected = projection_rows(run_dir, run, state, generated)
    return conversations, run, generated, projected
