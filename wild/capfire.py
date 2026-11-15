"""How often the released activation cap would bind on saved replies, per layer.

Usage: capfire.py chats RUN_DIR
       capfire.py jb PAIRS_JSONL REPLIES_JSONL OUT_JSONL [--n 150]
Env: AXIS_MODEL=qwen for Qwen 3 32B and its released cap (wcommon); default
     Llama 3.3 70B with the recommended cap.

chats: for each unit in RUN_DIR/drift.jsonl (a reply to user turn k of a chat in
RUN_DIR/selected.jsonl), runs the saved history through turn k plus the saved
reply, exactly as activations.py does, with the cap off. Appends one row per
unit to RUN_DIR/capfire.jsonl, so a rerun resumes.
jb: the same measurement on saved jailbreak replies (system prompt, question,
last saved reply) for the first N attacks of PAIRS_JSONL, as the reference;
capdiag.py measured these on 150 attacks (97% of reply tokens past the cap per
layer on Llama, 99% on Qwen).

Each row holds counts, not rates, so both a per-chat mean and capdiag.py's
token-pooled share can be computed (capfire_score.py):
- layers: the capped layers, in order; the per-layer lists follow this order.
- n_tok, n_prompt: reply tokens and the tokens before the reply.
- over: per layer, reply tokens whose projection on the cap's unit vector is
  past the cap (where the cap would act).
- excess: per layer, the summed amount past the cap over those tokens (what the
  cap would remove).
- margin: per layer, the summed projection minus cap over all reply tokens
  (negative when the reply sits below the cap on average).
- any_over: reply tokens past the cap at one or more layers; any_over / n_tok
  is activations.py's fire_frac.
- prompt_over: per layer, tokens before the reply that are past the cap.
Empty replies get a row with n_tok 0.
"""

import argparse
import json
import os

from huggingface_hub import hf_hub_download
import torch
import torch.nn.functional as F
from tqdm import tqdm
from wcommon import AXIS_REPO
from wcommon import CAPPING_EXPERIMENT
from wcommon import CAPPING_FILE
from wcommon import CHAT_KW
from wcommon import history
from wcommon import MODEL
from wcommon import read_jsonl

from assistant_axis import load_capping_config
from assistant_axis.internals import ConversationEncoder
from assistant_axis.internals import ProbingModel
from assistant_axis.internals import StopForward

# cuDNN's attention kernel crashed on large padded batches (2026-11-02:
# mha_graph.execute ... is_good()).
torch.backends.cuda.enable_cudnn_sdp(False)


def reply_span(spans, msgs):
    """[start, end) of the last message's tokens (the reply), or None."""
    last = sum(m["role"] != "system" for m in msgs) - 1
    span = next(
        (s for s in spans if s["turn"] == last and s["role"] == "assistant"),
        None,
    )
    if span is None or span["end"] <= span["start"]:
        return None
    return span["start"], span["end"]


def counts(excess, a, b):
    """Row fields from {layer: per-token projection minus cap} and the reply
    span [a, b).
    """
    order = sorted(excess)
    reply = torch.stack([excess[i][a:b] for i in order])  # layers x tokens.
    prompt = torch.stack([excess[i][:a] for i in order])
    return {
        "layers": order,
        "n_tok": b - a,
        "n_prompt": a,
        "over": (reply > 0).sum(1).tolist(),
        "excess": reply.clamp(min=0).sum(1).tolist(),
        "margin": reply.sum(1).tolist(),
        "any_over": int((reply > 0).any(0).sum()),
        "prompt_over": (prompt > 0).sum(1).tolist(),
    }


def units(args):
    """(key, messages ending in the saved reply) for every unit to measure."""
    if args.mode == "chats":
        convs = {
            c["hash"]: c
            for c in read_jsonl(os.path.join(args.run_dir, "selected.jsonl"))
        }
        for r in read_jsonl(os.path.join(args.run_dir, "drift.jsonl")):
            c = convs[r["hash"]]
            yield {"hash": r["hash"], "k": r["k"]}, history(
                c["users"], c["gpt"], r["k"], c.get("system")
            ) + [{"role": "assistant", "content": r["reply"]}]
    else:
        replies = {r["id"]: r["replies"][-1] for r in read_jsonl(args.replies)}
        for p in read_jsonl(args.pairs)[: args.n]:
            yield {"id": p["id"]}, [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["question"]},
                {"role": "assistant", "content": replies[p["id"]]},
            ]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    chats = sub.add_parser("chats")
    chats.add_argument("run_dir")
    jb = sub.add_parser("jb")
    jb.add_argument("pairs")
    jb.add_argument("replies")
    jb.add_argument("out")
    jb.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    path = (
        os.path.join(args.run_dir, "capfire.jsonl")
        if args.mode == "chats"
        else args.out
    )
    done = (
        {json.dumps({k: r[k] for k in ("hash", "k", "id") if k in r}) for r in read_jsonl(path)}
        if os.path.exists(path)
        else set()
    )
    todo = [(key, msgs) for key, msgs in units(args) if json.dumps(key) not in done]
    print(
        f"capfire {args.mode}: model {MODEL}, cap {CAPPING_FILE}"
        f" {CAPPING_EXPERIMENT}, {len(done)} done, {len(todo)} to do -> {path}",
        flush=True,
    )
    if not todo:
        return
    cfg = load_capping_config(
        hf_hub_download(AXIS_REPO, CAPPING_FILE, repo_type="dataset")
    )
    exp = next(e for e in cfg["experiments"] if e["id"] == CAPPING_EXPERIMENT)
    caps = {}  # layer -> (unit vector, cap); the cap acts where proj > cap.
    for it in exp["interventions"]:
        v = cfg["vectors"][it["vector"]]
        caps[v["layer"]] = (
            F.normalize(v["vector"].float(), dim=0),
            float(it["cap"]),
        )
    last_layer = max(caps)
    print(f"capped layers {sorted(caps)}", flush=True)
    pm = ProbingModel(MODEL)
    enc = ConversationEncoder(pm.tokenizer, MODEL)
    layers = pm.get_layers()
    excess = {}

    def hook_for(i):
        vec, cap = caps[i]

        def hook(module, inputs, output):
            h = (output[0] if isinstance(output, tuple) else output)[0].float()
            excess[i] = (h @ vec.to(h.device) - cap).cpu()
            if i == last_layer:
                raise StopForward()

        return hook

    handles = [layers[i].register_forward_hook(hook_for(i)) for i in sorted(caps)]
    try:
        with open(path, "a") as f:
            for key, msgs in tqdm(todo, mininterval=60):
                full_ids, spans = enc.build_turn_spans(msgs, **CHAT_KW)
                span = reply_span(spans, msgs)
                if span is None:  # Empty reply.
                    row = {**key, "n_tok": 0}
                else:
                    excess.clear()
                    try:
                        with torch.inference_mode():
                            pm.model(
                                torch.tensor(
                                    [full_ids[: span[1]]], device=pm.device
                                )
                            )
                    except StopForward:
                        pass
                    row = {**key, **counts(excess, *span)}
                f.write(json.dumps(row) + "\n")
                f.flush()
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
