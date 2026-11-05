"""Capability cost of the cap, the paper's benchmarks and sizes (Section 5.1.3),
with lm-eval (transformers).

Usage: python -m wild.capping.capability RUN_DIR Staged (user's call, 2026-11-03), uncapped vs
       capped: stage 1: IFEval 200, MMLU Pro 210 (15 per subject), GSM8K 200,
       EQ-Bench all 171. stage 2 (only if RUN_DIR/capability_stage2_approved
       exists): the paper's sizes, IFEval all 541, MMLU Pro 1400 (100 per
       subject), GSM8K 1000, EQ-Bench 171. Answers are cached per condition
       (lm-eval use_cache), so stage 2 reuses every stage-1 answer. lm-eval's
       limit takes the first N items per task, where the paper subsampled
       (method not stated).
Metrics: the paper doesn't name them. GSM8K uses lm-eval's flexible-extract
(last number in the reply): strict-match only accepts the dataset's "#### N"
format, which chat replies don't use (it scored 0.005 on replies that were
correct, 2026-11-03). Writes RUN_DIR/capability.json.
Paper, Llama 3.3 70B uncapped / capped: IFEval 0.93 / 0.93, MMLU Pro 0.71 /
0.70, GSM8K 0.89 / 0.89, EQ-Bench 83.1 / 84.1 (Figure 10). Approved 2026-11-02.
"""

import argparse
import contextlib
import json
import os

import lm_eval
from lm_eval.models.huggingface import HFLM
from wild.common.settings import capping_config
from wild.common.settings import CAPPING_EXPERIMENT
from wild.common.settings import load_probing_model

from assistant_axis import build_capping_steerer

TASKS = {
    "ifeval": "prompt_level_strict_acc,none",
    "mmlu_pro": "exact_match,custom-extract",
    "gsm8k": "exact_match,flexible-extract",
    "eq_bench": "eqbench,none",
}
METRIC_KEYS = {
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
    "exact_match",
    "eqbench",
    "percent_parseable",
}
STAGES = {
    "stage1": {
        "ifeval": 200,
        "mmlu_pro": 15,
        "gsm8k": 200,
        "eq_bench": None,
    },  # Per (sub)task.
    "paper": {"ifeval": None, "mmlu_pro": 100, "gsm8k": 1000, "eq_bench": None},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    if not os.path.exists(os.path.join(args.run_dir, "capability_approved")):
        print(
            "capability check on hold: benchmark set not approved yet (touch"
            " RUN_DIR/capability_approved)"
        )
        return
    path = os.path.join(args.run_dir, "capability.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    pm = load_probing_model()
    lm = HFLM(
        pretrained=pm.model, tokenizer=pm.tokenizer, batch_size=48
    )  # 16 left the H200s mostly idle.
    cfg = capping_config()
    for stage, limits in STAGES.items():
        if stage == "paper" and not os.path.exists(
            os.path.join(args.run_dir, "capability_stage2_approved")
        ):
            print("stage 2 (paper sizes) on hold until approved", flush=True)
            break
        res.setdefault(stage, {})
        for cond in ["none", "capped"]:
            res[stage].setdefault(cond, {})
            res[stage].setdefault(cond + "_stderr", {})
            res[stage].setdefault(cond + "_metric", {})
            # A saved score is reused only if it was computed with the current
            # metric (GSM8K's changed once).
            todo = [
                t
                for t in TASKS
                if t not in res[stage][cond]
                or res[stage][cond + "_metric"].get(t) != TASKS[t]
            ]
            if not todo:
                continue
            steer = (
                build_capping_steerer(pm.model, cfg, CAPPING_EXPERIMENT)
                if cond == "capped"
                else contextlib.nullcontext()
            )
            with steer:
                for (
                    t
                ) in (
                    todo
                # Saved after each benchmark, so a stop loses at most the one in
                # progress.
                ):
                    out = lm_eval.simple_evaluate(
                        model=lm,
                        tasks=[t],
                        limit=limits[t],
                        apply_chat_template=True,
                        fewshot_as_multiturn=True,
                        random_seed=0,
                        log_samples=True,
                        use_cache=os.path.join(
                            args.run_dir, f"lmeval_cache_{cond}"
                        ),
                    )
                    r = out["results"][t]
                    # Per-question scores, so capped vs uncapped can be compared
                    # as paired differences
                    # ("Adding Error Bars to Evals", Miller 2024). Keyed by
                    # (subtask, doc_id).
                    with open(
                        os.path.join(
                            args.run_dir, f"samples_{stage}_{cond}_{t}.jsonl"
                        ),
                        "w",
                    ) as f:
                        for sub, rows in out["samples"].items():
                            for x in rows:
                                f.write(
                                    json.dumps({
                                        "task": sub,
                                        "doc_id": x["doc_id"],
                                        "filter": x.get("filter"),
                                        **{
                                            k: x[k]
                                            for k in x
                                            if k in METRIC_KEYS
                                        },
                                    })
                                    + "\n"
                                )
                    m = TASKS[t]
                    res[stage][cond][t], res[stage][cond + "_stderr"][t] = r[
                        m
                    ], r.get(m.replace(",", "_stderr,"))
                    res[stage][cond + "_metric"][t] = m
                    print(stage, cond, t, res[stage][cond][t], flush=True)
                    json.dump(res, open(path + ".tmp", "w"), indent=1)
                    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
