"""Project drift replies onto the Assistant Axis and check where the released cap would act (HF, bf16).

Usage: activations.py RUN_DIR
For each unit in RUN_DIR/drift.jsonl, runs the GPT history through turn k plus Llama's reply k:
- proj: layer-40 output averaged over the reply tokens, projected onto the unit released axis.
- fire_frac: share of reply tokens where at least one capped layer (56-71 in the recommended
  setting) is past its cap. Read from the uncapped forward pass, so it says how often the cap would
  step in, not what happens after it does.
Appends to RUN_DIR/projections.jsonl, so a rerun resumes.
"""
import argparse, json, os
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from assistant_axis import load_axis, load_capping_config
from assistant_axis.internals import ConversationEncoder, ProbingModel, StopForward
from wcommon import (AXIS_FILE, AXIS_REPO, CAPPING_EXPERIMENT, CAPPING_FILE, LAYER, MODEL, history, read_jsonl)
# cuDNN's attention kernel crashed on large padded batches (2026-11-02: mha_graph.execute ... is_good()).
torch.backends.cuda.enable_cudnn_sdp(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    convs = {c["hash"]: c for c in read_jsonl(os.path.join(args.run_dir, "selected.jsonl"))}
    path = os.path.join(args.run_dir, "projections.jsonl")
    done = {(r["hash"], r["k"]) for r in read_jsonl(path)} if os.path.exists(path) else set()
    todo = [r for r in read_jsonl(os.path.join(args.run_dir, "drift.jsonl")) if (r["hash"], r["k"]) not in done]
    if not todo:
        print("activations: nothing left to project", flush=True)
        return
    pm = ProbingModel(MODEL)
    enc = ConversationEncoder(pm.tokenizer, MODEL)
    layers = pm.get_layers()
    ax = F.normalize(load_axis(hf_hub_download(AXIS_REPO, AXIS_FILE, repo_type="dataset"))[LAYER].float(), dim=0)

    cfg = load_capping_config(hf_hub_download(AXIS_REPO, CAPPING_FILE, repo_type="dataset"))
    exp = next(e for e in cfg["experiments"] if e["id"] == CAPPING_EXPERIMENT)
    caps = {}  # layer -> (unit vector, tau); the cap acts where the projection on the vector exceeds tau.
    for it in exp["interventions"]:
        v = cfg["vectors"][it["vector"]]
        caps[v["layer"]] = (F.normalize(v["vector"].float(), dim=0), float(it["cap"]))
    last_layer = max(caps)

    def run(msgs):
        full_ids, spans = enc.build_turn_spans(msgs)
        last = sum(m["role"] != "system" for m in msgs) - 1  # Span turn indices skip the system prompt.
        span = next((s for s in spans if s["turn"] == last and s["role"] == "assistant"), None)
        if span is None:  # Empty reply.
            return None, None
        a, b = span["start"], span["end"]
        got, fired = {}, []

        def hook_for(i):
            def hook(module, inputs, output):
                h = (output[0] if isinstance(output, tuple) else output)[0, a:b].float()
                if i == LAYER:
                    got["proj"] = float(h.mean(0) @ ax.to(h.device))
                if i in caps:
                    v, tau = caps[i]
                    fired.append((h @ v.to(h.device) > tau).cpu())
                if i == last_layer:
                    raise StopForward()
            return hook

        handles = [layers[i].register_forward_hook(hook_for(i)) for i in sorted(set(caps) | {LAYER})]
        try:
            with torch.inference_mode():
                pm.model(torch.tensor([full_ids[:b]], device=pm.device))
        except StopForward:
            pass
        finally:
            for hd in handles:
                hd.remove()
        return got["proj"], float(torch.stack(fired).any(0).float().mean())

    with open(path, "a") as f:
        for r in tqdm(todo, mininterval=60):
            c = convs[r["hash"]]
            msgs = history(c["users"], c["gpt"], r["k"], c.get("system")) + [{"role": "assistant", "content": r["reply"]}]
            proj, fire = run(msgs)
            f.write(json.dumps({"hash": r["hash"], "k": r["k"], "proj": proj, "fire_frac": fire}) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
