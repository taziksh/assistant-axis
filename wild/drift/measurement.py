"""Project frozen drift replies, validating provenance before every resume.

Usage: python -m wild.drift.measurement RUN_DIR
Writes layer-40 mean reply projections and uncapped cap-firing fractions.
Requires a completed protected run from generate.py --drift-only.
"""

import argparse
import math
from pathlib import Path

from . import state as run_state


def projection_runtime():
    import torch

    count = torch.cuda.device_count()
    if count < 1:
        raise ValueError("activation extraction requires CUDA GPUs")
    return {"device_map": "auto", "gpus": [torch.cuda.get_device_name(i) for i in range(count)]}


def load_projector(run):
    """Load the pinned tokenizer/model and verified released vector files."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from assistant_axis import load_axis, load_capping_config
    from assistant_axis.internals import ProbingModel

    torch.backends.cuda.enable_cudnn_sdp(False)
    artifact = run["artifacts"]["model"]
    axis_path = run_state.artifact_path(run["artifacts"]["axis"])
    cap_path = run_state.artifact_path(run["artifacts"]["capping"])
    tokenizer = AutoTokenizer.from_pretrained(artifact["repo_id"], revision=artifact["revision"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        artifact["repo_id"], revision=artifact["revision"], dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    pm = ProbingModel.from_existing(model, tokenizer, model_name=artifact["repo_id"])
    layer = run["config"]["layer"]
    raw_axis = load_axis(axis_path)[layer].float()
    if not torch.isfinite(raw_axis).all() or raw_axis.norm() <= 0:
        raise ValueError("released axis must be finite and nonzero")
    ax = F.normalize(raw_axis, dim=0)
    cfg = load_capping_config(cap_path)
    exp = next(e for e in cfg["experiments"] if e["id"] == run["config"]["capping_experiment"])
    caps = {}
    for intervention in exp["interventions"]:
        vector = cfg["vectors"][intervention["vector"]]
        raw = vector["vector"].float()
        tau = float(intervention["cap"])
        if not torch.isfinite(raw).all() or raw.norm() <= 0 or not math.isfinite(tau):
            raise ValueError("released cap must contain finite, nonzero vectors and finite thresholds")
        caps[vector["layer"]] = (F.normalize(raw, dim=0), tau)
    if not caps:
        raise ValueError("released capping experiment has no interventions")
    return exact_token_projector(pm, ax, caps, layer)


def exact_token_projector(pm, axis, caps, layer):
    """Project post-block residuals at the saved generated content positions."""
    import torch
    from assistant_axis.internals import StopForward

    layers = pm.get_layers()
    last_layer = max(layer, *caps)

    def project(response):
        prompt = response["prompt_token_ids"]
        full_ids = prompt + response["response_token_ids"]
        positions = [len(prompt) + i for i in run_state.content_positions(response)]
        if not positions:
            raise ValueError("reply has no content tokens to project")
        got, fired = {}, []

        def hook_for(index):
            def hook(module, inputs, output):
                h = (output[0] if isinstance(output, tuple) else output)[0, positions].float()
                if index == layer:
                    got["proj"] = float(h.mean(0) @ axis.to(h.device))
                if index in caps:
                    vector, threshold = caps[index]
                    fired.append((h @ vector.to(h.device) > threshold).cpu())
                if index == last_layer:
                    raise StopForward()
            return hook

        handles = [layers[i].register_forward_hook(hook_for(i)) for i in sorted(set(caps) | {layer})]
        try:
            with torch.inference_mode():
                # Tokens after the last measured position cannot affect it in
                # a causal model. Avoid retaining a KV cache during this pass.
                pm.model(torch.tensor([full_ids[:positions[-1] + 1]], device=pm.device), use_cache=False)
        except StopForward:
            pass
        finally:
            for handle in handles:
                handle.remove()
        return got["proj"], float(torch.stack(fired).any(0).float().mean())

    return project


def run_projection(run_dir):
    with run_state.run_lock(run_dir):
        if not (Path(run_dir) / "drift.state.json").exists():
            raise ValueError("run generate.py --drift-only before extracting protected projections")
        conversations, run = run_state.prepare_run(run_dir)
        drift_state = run_state.stage_state(run_dir, "drift", run)
        generated = run_state.generation_rows(run_dir, run, drift_state, conversations)
        state = run_state.stage_state(
            run_dir, "projections", run,
            {"drift_sha256": drift_state["output_sha256"], "drift_stage_id": drift_state["stage_id"]},
        )
        rows = run_state.projection_rows(run_dir, run, state, generated)
        if state["status"] == "complete":
            print(f"activations: verified {len(rows)} cached projections")
            return
        done = {run_state.row_key(row) for row in rows}
        todo = [row for row in generated if run_state.row_key(row) not in done]
        projector = None
        if any(run_state.content_positions(row) for row in todo):
            runtime = projection_runtime()
            if "runtime" in state and state["runtime"] != runtime:
                raise ValueError("extraction GPU configuration changed since the stage started")
            state["runtime"] = runtime
            run_state.save_record(Path(run_dir) / "projections.state.json", state)
            projector = load_projector(run)
        for response in todo:
            count = len(run_state.content_positions(response))
            row = {"hash": response["hash"], "k": response["k"], "sample_index": response["sample_index"],
                   "run_id": run["run_id"], "stage_id": state["stage_id"],
                   "response_token_count": count, "truncated": response["truncated"]}
            if count == 0:
                row.update(proj=None, fire_frac=None, reason="no_content_tokens")
            else:
                proj, fire = projector(response)
                if not math.isfinite(proj) or not math.isfinite(fire) or not 0 <= fire <= 1:
                    raise ValueError("extractor returned an invalid projection or cap firing fraction")
                row.update(proj=proj, fire_frac=fire)
            rows.append(run_state.append_projection(run_dir, row))
        run_state.complete_stage(run_dir, "projections", state, rows)
        run_state.projection_rows(run_dir, run, state, generated)
        print(f"activations: saved {len(rows)} projections")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    run_projection(args.run_dir)


if __name__ == "__main__":
    main()
