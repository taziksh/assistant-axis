"""Fig. 11 of the paper on Qwen 3 32B: Assistant Axis projection per turn of the
released insider-trading jailbreak, uncapped and capped.

Usage: fig11.py TRANSCRIPT_DIR OUT_JSONL [--rollouts N]
Scores the replies in each released transcript, then has Qwen answer the same
user turns itself N times per condition (temperature 0.7, 512 tokens, thinking
off), each rollout continuing from its own replies, and scores those. Projection:
layer 32, mean over a reply's tokens, in one pass of the whole conversation.
"""

import argparse
import contextlib
import json
import os

from huggingface_hub import hf_hub_download
import torch
import torch.nn.functional as F
from wcommon import AXIS_REPO
from wcommon import write_jsonl

from assistant_axis import build_capping_steerer
from assistant_axis import get_config
from assistant_axis import load_axis
from assistant_axis import load_capping_config
from assistant_axis.internals import ConversationEncoder
from assistant_axis.internals import ProbingModel
from assistant_axis.internals import StopForward

torch.backends.cuda.enable_cudnn_sdp(False)

MODEL = "Qwen/Qwen3-32B"
AXIS = "qwen-3-32b/assistant_axis.pt"
NEW_TOKENS = 512
FILES = {"none": "jailbreak_unsteered.json", "capped": "jailbreak_capped.json"}


def project(pm, enc, axis, layer, conv):
    """Per-reply projection for a conversation, in order."""
    ids, spans = enc.build_turn_spans(conv, enable_thinking=False)
    spans = [s for s in spans if s["role"] == "assistant"]
    got = {}

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        got["proj"] = [
            float(h[0, s["start"] : s["end"]].float().mean(0) @ axis.to(h.device))
            if s["end"] > s["start"]
            else None
            for s in spans
        ]
        raise StopForward()

    handle = pm.get_layers()[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            pm.model(torch.tensor([ids], device=pm.device))
    except StopForward:
        pass
    finally:
        handle.remove()
    return got["proj"]


def rollouts(pm, users, n):
    """n conversations over the same user turns, each from its own replies."""
    tok = pm.tokenizer
    tok.padding_side = "left"
    convs = [[] for _ in range(n)]
    for u in users:
        for c in convs:
            c.append({"role": "user", "content": u})
        texts = [
            tok.apply_chat_template(
                c,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for c in convs
        ]
        batch = tok(texts, return_tensors="pt", padding=True).to(pm.device)
        with torch.inference_mode():
            gen = pm.model.generate(
                **batch,
                max_new_tokens=NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=1.0,
                top_k=0,
                pad_token_id=tok.pad_token_id,
            )
        for c, g in zip(convs, gen[:, batch["input_ids"].shape[1] :]):
            c.append({
                "role": "assistant",
                "content": tok.decode(g, skip_special_tokens=True),
            })
    return convs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript_dir")
    ap.add_argument("out")
    ap.add_argument("--rollouts", type=int, default=10)
    args = ap.parse_args()
    torch.manual_seed(0)
    cfg = get_config(MODEL)
    layer = cfg["target_layer"]
    axis = F.normalize(
        load_axis(hf_hub_download(AXIS_REPO, AXIS, repo_type="dataset"))[
            layer
        ].float(),
        dim=0,
    )
    cap = load_capping_config(
        hf_hub_download(AXIS_REPO, cfg["capping_config"], repo_type="dataset")
    )
    pm = ProbingModel(MODEL)
    enc = ConversationEncoder(pm.tokenizer, MODEL)
    rows = []
    for cond, name in FILES.items():
        with open(os.path.join(args.transcript_dir, name)) as f:
            conv = json.load(f)["conversation"]
        for t, p in enumerate(project(pm, enc, axis, layer, conv)):
            rows.append({
                "source": "released",
                "condition": cond,
                "rollout": None,
                "turn": t + 1,
                "proj": p,
                "reply": conv[2 * t + 1]["content"],
            })
        users = [m["content"] for m in conv if m["role"] == "user"]
        with (
            build_capping_steerer(pm.model, cap, cfg["capping_experiment"])
            if cond == "capped"
            else contextlib.nullcontext()
        ):
            ours = rollouts(pm, users, args.rollouts)
        for r, c in enumerate(ours):
            for t, p in enumerate(project(pm, enc, axis, layer, c)):
                rows.append({
                    "source": "ours",
                    "condition": cond,
                    "rollout": r,
                    "turn": t + 1,
                    "proj": p,
                    "reply": c[2 * t + 1]["content"],
                })
        write_jsonl(args.out, rows)
        print(cond, "done", flush=True)


if __name__ == "__main__":
    main()
