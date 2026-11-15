"""How often the released activation cap binds, on saved jailbreak replies.

Usage: capdiag.py PAIRS_JSONL REPLIES_DIR OUT_JSON [--n N]
Env: CAP_MODEL, CAP_FILE, CAP_EXP as in capping.py (default Llama 3.3 70B).
Teacher-forces the first N prompts with each saved reply set
(REPLIES_DIR/shah_{none,capped}.jsonl) and, at every capped layer, measures
each token's projection onto the cap's unit vector against its cap: the share
of reply and prompt tokens above it and the mean excess. Run once without the
cap (how often it would bind) and once inside it (a check that it clamps: the
share above should be 0).
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

import capping  # Env overrides for model and cap.
from wcommon import AXIS_REPO
from wcommon import read_jsonl

from assistant_axis import build_capping_steerer
from assistant_axis import load_capping_config
from assistant_axis.internals import ProbingModel

torch.backends.cuda.enable_cudnn_sdp(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("replies_dir")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    pairs = read_jsonl(args.pairs)[: args.n]
    cfg = load_capping_config(
        hf_hub_download(AXIS_REPO, capping.CAPPING_FILE, repo_type="dataset")
    )
    exp = next(e for e in cfg["experiments"] if e["id"] == capping.CAPPING_EXPERIMENT)
    caps = {}  # layer -> (unit vector, cap)
    for iv in exp["interventions"]:
        v = cfg["vectors"][iv["vector"]]
        caps[v["layer"]] = (F.normalize(v["vector"].float(), dim=0), float(iv["cap"]))
    print(f"model {capping.MODEL}, {capping.CAPPING_EXPERIMENT}, layers {sorted(caps)}", flush=True)
    pm = ProbingModel(capping.MODEL)
    tok = pm.tokenizer
    layers = pm.get_layers()
    got = {}

    def hook_for(layer):
        vec, cap = caps[layer]

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            got[layer] = (h[0].float() @ vec.to(h.device)) - cap  # Excess per token.

        return hook

    summary = {"model": capping.MODEL, "experiment": capping.CAPPING_EXPERIMENT, "n": len(pairs), "runs": {}}
    for reply_set in ("none", "capped"):
        replies = {r["id"]: r["replies"][-1] for r in read_jsonl(os.path.join(args.replies_dir, f"shah_{reply_set}.jsonl"))}
        for mode in ("off", "on"):
            stats = {L: {"reply_over": 0, "reply_excess": 0.0, "reply_n": 0, "prompt_over": 0, "prompt_n": 0} for L in caps}
            steer = build_capping_steerer(pm.model, cfg, capping.CAPPING_EXPERIMENT) if mode == "on" else None
            if steer:
                steer.__enter__()  # Registered first, so the measuring hooks see capped outputs.
            handles = [layers[L].register_forward_hook(hook_for(L)) for L in caps]
            try:
                for p in pairs:
                    msgs = [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["question"]}]
                    prompt = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **capping.TEMPLATE_KW), add_special_tokens=False).input_ids
                    reply = tok(replies[p["id"]], add_special_tokens=False).input_ids
                    if not reply:
                        continue
                    with torch.inference_mode():
                        pm.model(torch.tensor([prompt + reply], device=pm.device))
                    k = len(prompt)
                    for L, ex in got.items():
                        s = stats[L]
                        s["reply_over"] += int((ex[k:] > 0).sum())
                        s["reply_excess"] += float(ex[k:].clamp(min=0).sum())
                        s["reply_n"] += len(reply)
                        s["prompt_over"] += int((ex[:k] > 0).sum())
                        s["prompt_n"] += k
            finally:
                for h in handles:
                    h.remove()
                if steer:
                    steer.__exit__()
            per_layer = {
                L: {
                    "reply_frac_over": s["reply_over"] / max(s["reply_n"], 1),
                    "reply_mean_excess": s["reply_excess"] / max(s["reply_n"], 1),
                    "prompt_frac_over": s["prompt_over"] / max(s["prompt_n"], 1),
                }
                for L, s in stats.items()
            }
            summary["runs"][f"replies_{reply_set}/cap_{mode}"] = per_layer
            avg = sum(x["reply_frac_over"] for x in per_layer.values()) / len(per_layer)
            print(f"replies {reply_set}, cap {mode}: mean share of reply tokens over cap {avg:.3f}", flush=True)
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
