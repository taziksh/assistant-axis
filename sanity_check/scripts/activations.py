"""Project scored replies onto the released Assistant Axis (HF transformers,
bf16).

Usage: activations.py OUT_DIR [--model M --layer L --axis PATH]  (overrides are
       for local tests)
Reads OUT_DIR/histories.jsonl and OUT_DIR/scored.jsonl. For each scored sample,
runs the history plus the reply through the model, means the layer-L output over
the reply tokens (spans from ConversationEncoder.build_turn_spans), and projects
onto the unit axis. Also the teacher-forced
readout: GPT's original reply to turn k, as writer "gpt_tf".
Appends to OUT_DIR/projections.jsonl, so a rerun resumes.
"""

import argparse
import json
import os
import sys

from huggingface_hub import hf_hub_download
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import AXIS_FILE
from common import AXIS_REPO
from common import context
from common import history
from common import LAYER
from common import MODEL
from common import read_jsonl
from common import scored_turns

from assistant_axis import load_axis
from assistant_axis.internals import ConversationEncoder
from assistant_axis.internals import ProbingModel
from assistant_axis.internals import StopForward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument(
        "--axis",
        default=None,
        help="local .pt of shape (n_layers, hidden); default: released axis",
    )
    args = ap.parse_args()

    pm = ProbingModel(args.model)
    enc = ConversationEncoder(pm.tokenizer, args.model)
    axis = load_axis(
        args.axis or hf_hub_download(AXIS_REPO, AXIS_FILE, repo_type="dataset")
    )
    ax = F.normalize(axis[args.layer].float(), dim=0).to(pm.device)
    layer = pm.get_layers()[args.layer]

    def project(msgs):
        """Projection of the mean layer output over the last message (an
        assistant reply).
        """
        full_ids, spans = enc.build_turn_spans(msgs)
        last = len([m for m in msgs if m["role"] != "system"]) - 1
        span = next(
            (
                s
                for s in spans
                if s["turn"] == last and s["role"] == "assistant"
            ),
            None,
        )
        if span is None:  # Empty reply.
            return None
        got = {}

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            got["mean"] = h[0, span["start"] : span["end"]].float().mean(0)
            raise StopForward()

        handle = layer.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                pm.model(
                    torch.tensor([full_ids[: span["end"]]], device=pm.device)
                )
        except StopForward:
            pass
        finally:
            handle.remove()
        return float(
            got["mean"].to(ax.device) @ ax
        )  # The layer may sit on another GPU.

    convs = {
        c["hash"]: c
        for c in read_jsonl(os.path.join(args.out_dir, "histories.jsonl"))
    }
    work = []  # (key, messages)
    for r in read_jsonl(os.path.join(args.out_dir, "scored.jsonl")):
        c = convs[r["hash"]]
        base = context(c, r["k"], r["writer"])
        for i, s in enumerate(r["samples"]):
            work.append((
                (r["hash"], r["k"], r["writer"], i),
                base + [{"role": "assistant", "content": s}],
            ))
    # Teacher-forced: GPT's original reply to turn k after GPT's history (writer
    # "gpt_tf").
    for c in convs.values():
        for k in scored_turns(len(c["users"])):
            msgs = history(c["users"], c["replies"]["gpt"], k) + [
                {"role": "assistant", "content": c["replies"]["gpt"][k - 1]}
            ]
            work.append(((c["hash"], k, "gpt_tf", 0), msgs))

    path = os.path.join(args.out_dir, "projections.jsonl")
    done = set()
    if os.path.exists(path):
        done = {
            (r["hash"], r["k"], r["writer"], r["sample"])
            for r in read_jsonl(path)
        }
    with open(path, "a") as f:
        for key, msgs in tqdm(
            [w for w in work if w[0] not in done], mininterval=60
        ):
            h, k, w, i = key
            f.write(
                json.dumps({
                    "hash": h,
                    "k": k,
                    "writer": w,
                    "sample": i,
                    "proj": project(msgs),
                })
                + "\n"
            )
            f.flush()


if __name__ == "__main__":
    main()
