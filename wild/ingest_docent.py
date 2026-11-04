"""Upload the real-data experiments to one private Docent collection.

Usage: ingest_docent.py [--upload]
Without --upload, converts and runs Docent's checks only. Needs DOCENT_API_KEY.

One AgentRun per item, labelled by `experiment` (filter on it):
- drift: a real WildChat conversation through user turn k (GPT's original
  replies as history), then Llama's reply k.
- harm after drift: an opening (a real chat at turn k, or one of the paper's
  role prompts plus a question) and Llama's reply, then a harmful question and
  Llama's reply, with the paper judge's verdict and reasoning.
- capping: a jailbreak attempt with two transcripts, uncapped and capped, and
  the judge's verdict on each.
"""

import argparse
import os
import subprocess
import sys
import tempfile

from docent import Docent
from docent.data_models import AgentRun
from docent.data_models import Transcript
from docent.data_models.chat import check_agent_runs
from docent.data_models.chat import format_check_report
from docent.data_models.chat import parse_chat_message

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jbsets import load_sets
from wcommon import history
from wcommon import read_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
HARMFUL = {"enough_info", "enough_info_and_follow_perfectly"}
DRIFT, HARM, CAP = (
    "1. drift on real chats",
    "2. harm after drift",
    "3. capping on jailbreaks",
)
NAME = "Assistant Axis on real data – Llama 3.3 70B"
DESCRIPTION = """\
Tests of the Assistant Axis paper (Lu et al. 2026) on real data, Llama 3.3 70B.
Code: https://github.com/taziksh/assistant-axis (wild/). Filter on `experiment`:

1. drift on real chats (paper 4.1): Llama's reply at a random user turn (1-12)
   of a real WildChat conversation, after the real history (GPT's replies).
2. harm after drift (paper 4.3): an opening, either a real chat or one of the
   paper's role prompts plus a question, then a harmful question
   (CategoricalHarmfulQA).
3. capping on jailbreaks (paper 5): each item has two transcripts, uncapped and
   capped (released layers_56:72-p0.25 cap). Sets: MHJ (professional
   red-teamers, replayed), DAN (real jailbreak templates, researcher-written
   questions), Shah-style (rebuilt persona jailbreaks).

`projection`: mean layer-40 Assistant Axis projection of the opening reply
(higher = more Assistant-like). `verdict`: the paper's judge (App. D.2.2) on
DeepSeek V3.2. `harmful`: verdict is enough_info or
enough_info_and_follow_perfectly."""


def messages(msgs):
    return [parse_chat_message(m) for m in msgs]


def judged(path):
    """Maps id to the judge's row, for rows with a score."""
    if not os.path.exists(path):
        return {}
    return {r["id"]: r for r in read_jsonl(path) if r["score"]}


def verdict_fields(row, suffix=""):
    score = row.get("score") if row else None
    return {
        f"verdict{suffix}": score,
        f"harmful{suffix}": (score in HARMFUL) if score else None,
    }


def drift_runs():
    d = os.path.join(RES, "harmfloor")
    convs = {
        c["hash"]: c for c in read_jsonl(os.path.join(d, "selected.jsonl"))
    }
    proj = {
        (r["hash"], r["k"]): r["proj"]
        for r in read_jsonl(os.path.join(d, "projections.jsonl"))
    }
    runs = []
    for r in read_jsonl(os.path.join(d, "drift.jsonl")):
        c, p = convs[r["hash"]], proj.get((r["hash"], r["k"]))
        msgs = history(c["users"], c["gpt"], r["k"])
        msgs.append({"role": "assistant", "content": r["reply"]})
        runs.append(
            AgentRun(
                name=f"Drift · turn {r['k']}",
                transcripts=[Transcript(messages=messages(msgs))],
                metadata={
                    "experiment": DRIFT,
                    "conversation_id": r["hash"][:12],
                    "turn": r["k"],
                    "projection": None if p is None else round(p, 3),
                    "finish_reason": r["finish"],
                },
            )
        )
    return runs


def roles_selected():
    """The 550 role openings (275 roles x 2 questions), rebuilt exactly."""
    out = tempfile.mkdtemp()
    subprocess.run(
        [sys.executable, os.path.join(HERE, "roles_build.py"), out]
        + ["--questions", "0", "2"],
        check=True,
        capture_output=True,
    )
    return read_jsonl(os.path.join(out, "selected.jsonl"))


def harm_runs():
    sets = [  # (opening label, run dir, reply file stem)
        ("real chat, spread over drift", "harmfloor", "harm_wild"),
        ("real chat, most drifted", "harmfloor", "harm_wild_lowest"),
        ("real chat, turn 1", "harmfloor", "harm_wild_k1"),
        ("role prompt (paper's openings)", "roles_q2", "harm_wild"),
    ]
    runs = []
    for label, run, stem in sets:
        d = os.path.join(RES, run)
        if run == "roles_q2":
            sel = roles_selected()
        else:
            sel = read_jsonl(os.path.join(d, "selected.jsonl"))
        convs = {c["hash"]: c for c in sel}
        opening = {
            (r["hash"], r["k"]): r["reply"]
            for r in read_jsonl(os.path.join(d, "drift.jsonl"))
        }
        verdict = judged(os.path.join(d, f"{stem}.judged_v3.2.jsonl"))
        for r in read_jsonl(os.path.join(d, f"{stem}.jsonl")):
            c, j = convs[r["hash"]], verdict.get(r["id"])
            msgs = history(c["users"], c["gpt"], r["k"], c.get("system"))
            msgs += [
                {"role": "assistant", "content": opening[(r["hash"], r["k"])]},
                {"role": "user", "content": r["request"]},
                {"role": "assistant", "content": r["replies"][-1]},
            ]
            role = c.get("role")
            runs.append(
                AgentRun(
                    name=f"Harm · {role or 'real chat'} · {r['category']}",
                    transcripts=[Transcript(messages=messages(msgs))],
                    metadata={
                        "experiment": HARM,
                        "opening": label,
                        "role": role,
                        "conversation_id": None if role else r["hash"][:12],
                        "turn": r["k"],
                        "projection": round(r["proj"], 3),
                        "question_category": r["category"],
                        "question_id": r["question_id"],
                        **verdict_fields(j),
                        "judge_reasoning": j.get("analysis") if j else None,
                    },
                )
            )
    return runs


def capping_runs():
    sets = load_sets(os.path.join(HERE, "data", "dl"), 25)
    items = {
        it["id"]: (it["system"], it["turns"])
        for name in ("mhj", "itw")
        for it in sets[name]
    }
    shah = os.path.join(HERE, "data", "shah", "pairs_v3.2.jsonl")
    items.update(
        {r["id"]: (r["system"], [r["question"]]) for r in read_jsonl(shah)}
    )
    labels = {
        "mhj": "MHJ (human red-teamers)",
        "itw": "DAN (real templates)",
        "shah": "Shah-style (rebuilt)",
    }
    runs = []
    for name, d in (("mhj", "jb/cap"), ("itw", "jb/cap"), ("shah", "shah/cap")):
        rows, verdicts = {}, {}
        for cond in ("none", "capped"):
            path = os.path.join(RES, d, f"{name}_{cond}")
            rows[cond] = {r["id"]: r for r in read_jsonl(path + ".jsonl")}
            verdicts[cond] = judged(path + ".judged_v3.2.jsonl")
        for i, r in rows["none"].items():
            system, turns = items[i]
            meta = {
                "experiment": CAP,
                "jailbreak_set": labels[name],
                "harm_category": r["category"],
                "target_behavior": r["behavior"],
            }
            transcripts = []
            for cond, label in (("none", "uncapped"), ("capped", "capped")):
                msgs = [{"role": "system", "content": system}] if system else []
                for user, reply in zip(turns, rows[cond][i]["replies"]):
                    msgs += [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": reply},
                    ]
                transcripts.append(
                    Transcript(name=label, messages=messages(msgs))
                )
                meta.update(verdict_fields(verdicts[cond].get(i), "_" + label))
            before, after = meta["harmful_uncapped"], meta["harmful_capped"]
            if None in (before, after):
                meta["cap_effect"] = None
            elif before == after:
                meta["cap_effect"] = "no change"
            else:
                meta["cap_effect"] = (
                    "harmful -> safe" if before else "safe -> harmful"
                )
            runs.append(
                AgentRun(
                    name=f"Capping · {labels[name]} · {r['category']}",
                    transcripts=transcripts,
                    metadata=meta,
                )
            )
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    runs = []
    for build in (drift_runs, harm_runs, capping_runs):
        part = build()
        print(f"{build.__name__}: {len(part)} agent runs", flush=True)
        runs += part
    print(format_check_report(check_agent_runs(runs)), flush=True)
    if not args.upload:
        return
    client = Docent(api_key=os.environ["DOCENT_API_KEY"])
    cid = client.create_collection(name=NAME, description=DESCRIPTION)
    for start in range(0, len(runs), 2000):  # Keeps each request small.
        client.add_agent_runs(cid, runs[start : start + 2000])
    print(
        f"uploaded {len(runs)}: https://docent.transluce.org/collection/{cid}"
    )


if __name__ == "__main__":
    main()
