"""Real jailbreaks against Llama 3.3 70B with and without the released
activation cap (HF, greedy).

Usage: capping.py RUN_DIR DL_DIR [--itw N] [--only SET:COND] [--shah
       PAIRS_JSONL]
Sets: itw and mhj from jbsets.py (DL_DIR holds the downloaded files, see
run.sh); or, with --shah, only the Shah-style persona set from shah_gen.py
(persona as system prompt, question as the user turn). Llama answers each user
turn in order after its own earlier replies; the last reply is judged
(judge_paper.py).
Conditions: none, and capped with the recommended released setting
(CAPPING_EXPERIMENT).
Writes RUN_DIR/cap/{set}_{condition}.jsonl; finished files are skipped. MHJ runs
first; its uncapped replies are also projected onto the axis
(RUN_DIR/cap/mhj_none_proj.jsonl) for the harm test.
"""

import argparse
import json
import os

from huggingface_hub import hf_hub_download
from jbsets import load_sets
import torch
import torch.nn.functional as F
from tqdm import tqdm
from wcommon import AXIS_FILE
from wcommon import AXIS_REPO
from wcommon import CAPPING_EXPERIMENT
from wcommon import CAPPING_FILE
from wcommon import HARM_NEW_TOKENS
from wcommon import LAYER
from wcommon import MODEL
from wcommon import read_jsonl
from wcommon import write_jsonl

from assistant_axis import build_capping_steerer
from assistant_axis import load_axis
from assistant_axis import load_capping_config
from assistant_axis.internals import ConversationEncoder
from assistant_axis.internals import ProbingModel
from assistant_axis.internals import StopForward

# cuDNN's attention kernel crashed on large padded batches (2026-11-02:
# mha_graph.execute ... is_good()).
torch.backends.cuda.enable_cudnn_sdp(False)

# Prompt + new tokens per batch; KV cache is 0.33 MB/token and ~9 GB per GPU is
# free on 2x H100.
TOKEN_BUDGET = 20000
MAX_BATCH = 48
MAX_PROMPT = 6000


def batched_generate(pm, prompts):
    """Greedy replies for a list of token-id lists; batches sorted by length
    under TOKEN_BUDGET.
    """
    tok = pm.tokenizer
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out, i = [None] * len(prompts), 0
    bar = tqdm(total=len(prompts), mininterval=60)
    while i < len(order):
        longest = len(prompts[order[min(len(order) - 1, i + MAX_BATCH - 1)]])
        bs = max(1, min(MAX_BATCH, TOKEN_BUDGET // (longest + HARM_NEW_TOKENS)))
        idx = order[i : i + bs]
        batch = tok.pad(
            {"input_ids": [prompts[k] for k in idx]}, return_tensors="pt"
        ).to(pm.device)
        with torch.inference_mode():
            gen = pm.model.generate(
                **batch,
                max_new_tokens=HARM_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        for k, g in zip(idx, gen[:, batch["input_ids"].shape[1] :]):
            out[k] = tok.decode(g, skip_special_tokens=True)
        i += bs
        bar.update(bs)
    return out


def run_set(pm, items, ckpt=None):
    """Answers every user turn in order; returns the per-item list of replies.
    With ckpt, saves the

    replies after each turn and resumes from them.
    """
    tok = pm.tokenizer
    replies = {it["id"]: [] for it in items}
    if ckpt and os.path.exists(ckpt):
        replies.update(json.load(open(ckpt)))
    for t in range(max(len(it["turns"]) for it in items)):
        if all(
            len(replies[it["id"]]) > t for it in items if t < len(it["turns"])
        ):
            continue  # Turn already done (checkpoint).
        live, prompts = [], []
        for it in items:
            if t >= len(it["turns"]):
                continue
            msgs = (
                [{"role": "system", "content": it["system"]}]
                if it["system"]
                else []
            )
            for u, a in zip(it["turns"][:t], replies[it["id"]]):
                msgs += [
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ]
            msgs.append({"role": "user", "content": it["turns"][t]})
            ids = tok(
                tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                ),
                add_special_tokens=False,
            ).input_ids
            if len(ids) > MAX_PROMPT:
                ids = (
                    ids[:1] + ids[-(MAX_PROMPT - 1) :]
                )  # Keep BOS and the most recent context.
            live.append(it)
            prompts.append(ids)
        print(f"  turn {t + 1}: {len(prompts)} prompts", flush=True)
        for it, r in zip(live, batched_generate(pm, prompts)):
            replies[it["id"]].append(r)
        if ckpt:
            json.dump(replies, open(ckpt + ".tmp", "w"))
            os.replace(ckpt + ".tmp", ckpt)
    return replies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("dl_dir")
    ap.add_argument(
        "--itw",
        type=int,
        default=25,
        help="DAN pairs per forbidden-question category",
    )
    ap.add_argument(
        "--only", default=None, help="e.g. mhj:none to run one set/condition"
    )
    ap.add_argument(
        "--shah",
        default=None,
        help=(
            "Shah-style pairs (shah_gen.py); runs only that set, none then"
            " capped"
        ),
    )
    args = ap.parse_args()
    out_dir = os.path.join(args.run_dir, "cap")
    os.makedirs(out_dir, exist_ok=True)
    if args.shah:
        # Persona prompt as system prompt, question as the user turn (both
        # papers). The judge's request is the
        # whole jailbreak input; behavior is the harm category.
        sets = {
            "shah": [
                {
                    "id": r["id"],
                    "system": r["system"],
                    "turns": [r["question"]],
                    "behavior": r["category"],
                    "context": None,
                    "category": r["category"],
                    "request": r["system"] + "\n\n" + r["question"],
                }
                for r in read_jsonl(args.shah)
            ]
        }
    else:
        sets = load_sets(args.dl_dir, args.itw)
    pm = ProbingModel(MODEL)
    cfg = load_capping_config(
        hf_hub_download(AXIS_REPO, CAPPING_FILE, repo_type="dataset")
    )
    # MHJ first: its uncapped replies are also the harm test (paper 4.3 on real
    # attacks).
    # ToxicChat is loaded but not run: it has no target behavior for the judge
    # (dropped 2026-11-02).
    pairs = (
        [("shah", "none"), ("shah", "capped")]
        if args.shah
        else [
            ("mhj", "none"),
            ("mhj", "capped"),
            ("itw", "none"),
            ("itw", "capped"),
        ]
    )
    for name, cond in pairs:
        if args.only and f"{name}:{cond}" != args.only:
            continue
        items = sets[name]
        path = os.path.join(out_dir, f"{name}_{cond}.jsonl")
        if not os.path.exists(path):
            print(f"{name} / {cond}: {len(items)} items", flush=True)
            if cond == "capped":
                with build_capping_steerer(pm.model, cfg, CAPPING_EXPERIMENT):
                    replies = run_set(pm, items, path + ".ckpt")
            else:
                replies = run_set(pm, items, path + ".ckpt")
            write_jsonl(
                path,
                [
                    {
                        **{k: v for k, v in it.items() if k != "turns"},
                        "n_turns": len(it["turns"]),
                        "condition": cond,
                        "replies": replies[it["id"]],
                    }
                    for it in items
                ],
            )
        proj_path = os.path.join(out_dir, "mhj_none_proj.jsonl")
        if (name, cond) == ("mhj", "none") and not os.path.exists(proj_path):
            print("projecting MHJ replies", flush=True)
            replies = {r["id"]: r["replies"] for r in read_jsonl(path)}
            write_jsonl(
                proj_path,
                [
                    {
                        "id": it["id"],
                        "proj": project_replies(pm, it, replies[it["id"]]),
                    }
                    for it in tqdm(items, mininterval=60)
                ],
            )


def project_replies(pm, it, replies):
    """Layer-40 projection of each of Llama's replies, averaged over its tokens,
    in one forward pass of

    the whole conversation. (Generation cut prompts over MAX_PROMPT tokens to their
    end; this pass sees the full conversation, so the few very long attacks differ
    slightly.)
    """
    global _enc, _ax
    if "_enc" not in globals():
        _enc = ConversationEncoder(pm.tokenizer, MODEL)
        _ax = F.normalize(
            load_axis(
                hf_hub_download(AXIS_REPO, AXIS_FILE, repo_type="dataset")
            )[LAYER].float(),
            dim=0,
        )
    msgs = [{"role": "system", "content": it["system"]}] if it["system"] else []
    for u, a in zip(it["turns"], replies):
        msgs += [
            {"role": "user", "content": u},
            {"role": "assistant", "content": a},
        ]
    full_ids, spans = _enc.build_turn_spans(msgs)
    # Turn index counts non-system messages, so reply j is turn 2j + 1; empty
    # replies have no span.
    by_turn = {sp["turn"]: sp for sp in spans if sp["role"] == "assistant"}
    spans = [by_turn.get(2 * j + 1) for j in range(len(replies))]
    got = {}

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        got["proj"] = [
            float(
                h[0, sp["start"] : sp["end"]].float().mean(0) @ _ax.to(h.device)
            )
            if sp and sp["end"] > sp["start"]
            else None
            for sp in spans
        ]
        raise StopForward()

    handle = pm.get_layers()[LAYER].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            pm.model(torch.tensor([full_ids], device=pm.device))
    except StopForward:
        pass
    finally:
        handle.remove()
    return got["proj"]


if __name__ == "__main__":
    main()
