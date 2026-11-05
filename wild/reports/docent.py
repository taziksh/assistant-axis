"""Upload the real-data experiments to private Docent collections.

Usage: python -m wild.reports.docent [--only drift|harm|capping ...] [--upload]
Without --upload, converts and runs Docent's checks only. Needs DOCENT_API_KEY.

Collections (one AgentRun per item):
- drift: a real WildChat conversation through user turn k (GPT's original
  replies as history), then Llama's reply k. Metadata: turn, projection.
- harm: an opening (real chat at turn k, or a role prompt plus a question) and
  Llama's reply, then one harmful question and Llama's reply, with the paper
  judge's verdict and reasoning.
- capping: a jailbreak attempt with two transcripts, uncapped and capped, and
  the judge's verdict on each.
"""

import argparse
import os

from wild.common.paths import REPO_ROOT, WILD_ROOT
import subprocess
import sys
import tempfile

from docent import Docent
from docent.data_models import AgentRun
from docent.data_models import Transcript
from docent.data_models.chat import check_agent_runs
from docent.data_models.chat import format_check_report
from docent.data_models.chat import parse_chat_message

from wild.capping.datasets import load_sets
from wild.common.settings import history
from wild.common.settings import read_jsonl

HERE = WILD_ROOT
RES = os.path.join(HERE, "results")
HARMFUL = {"enough_info", "enough_info_and_follow_perfectly"}
COLLECTIONS = {
    "drift": (
        "AA real-chat drift – Llama 3.3 70B",
        (
            "Llama 3.3 70B's reply at a random user turn k (1-12) of a real"
            " WildChat-4.8M conversation, after the real history (GPT's"
            " original replies). 'projection' is the mean layer-40 Assistant"
            " Axis projection of the reply (higher = more Assistant-like)."
        ),
    ),
    "harm": (
        "AA harm after drift – Llama 3.3 70B",
        (
            "Paper Sec. 4.3 on real data. An opening (a real WildChat chat at"
            " turn k, or one of the paper's role prompts plus an unrelated"
            " question) and Llama's reply, then a harmful question"
            " (CategoricalHarmfulQA) and Llama's reply. 'verdict' is the"
            " paper's judge (App. D.2.2) on DeepSeek V3.2; harmful ="
            " enough_info or enough_info_and_follow_perfectly."
        ),
    ),
    "capping": (
        "AA capping on jailbreaks – Llama 3.3 70B",
        (
            "Paper Sec. 5. Each run is one jailbreak attempt with two"
            " transcripts: uncapped and capped (released layers_56:72-p0.25"
            " cap, greedy). set: MHJ (Scale AI's human red-teaming attacks,"
            " replayed), DAN (in-the-wild jailbreak templates with"
            " researcher-written questions), or Shah et al. (regenerated:"
            " persona jailbreaks written with Shah et al.'s method; theirs"
            " were not released). category: from each source dataset."
            " verdict_*: the paper's judge (App. D.2.2, its 9 categories) on"
            " DeepSeek V3.2."
        ),
    ),
}


def messages(msgs):
    return [parse_chat_message(m) for m in msgs]


def judged(path):
    """id -> judge row, for rows with a score."""
    return (
        {r["id"]: r for r in read_jsonl(path) if r["score"]}
        if os.path.exists(path)
        else {}
    )


def drift_runs():
    d = os.path.join(RES, "harmfloor")
    convs = {
        c["hash"]: c for c in read_jsonl(os.path.join(d, "selected.jsonl"))
    }
    proj = {
        (r["hash"], r["k"]): r
        for r in read_jsonl(os.path.join(d, "projections.jsonl"))
    }
    runs = []
    for r in read_jsonl(os.path.join(d, "drift.jsonl")):
        c, p = convs[r["hash"]], proj.get((r["hash"], r["k"]), {})
        msgs = history(c["users"], c["gpt"], r["k"]) + [
            {"role": "assistant", "content": r["reply"]}
        ]
        runs.append(
            AgentRun(
                transcripts=[Transcript(messages=messages(msgs))],
                metadata={
                    "conversation": r["hash"][:12],
                    "turn": r["k"],
                    "projection": p.get("proj"),
                    "cap_fire_frac": p.get("fire_frac"),
                    "finish_reason": r["finish"],
                },
            )
        )
    return runs


def roles_selected():
    """roles_q2's openings (275 roles x the first 2 questions), rebuilt deterministically."""
    out = tempfile.mkdtemp()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "wild.harm.roles",
            out,
            "--questions",
            "0",
            "2",
        ],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )
    return read_jsonl(os.path.join(out, "selected.jsonl"))


def harm_runs():
    sets = [  # (opening type, run dir, reply file stem)
        ("real (spread over drift)", "harmfloor", "harm_wild"),
        ("real (most drifted)", "harmfloor", "harm_wild_lowest"),
        ("real (turn 1)", "harmfloor", "harm_wild_k1"),
        ("role prompt (paper's openings)", "roles_q2", "harm_wild"),
    ]
    runs = []
    for kind, run, stem in sets:
        d = os.path.join(RES, run)
        sel = (
            roles_selected()
            if run == "roles_q2"
            else read_jsonl(os.path.join(d, "selected.jsonl"))
        )
        convs = {c["hash"]: c for c in sel}
        opening = {
            (r["hash"], r["k"]): r["reply"]
            for r in read_jsonl(os.path.join(d, "drift.jsonl"))
        }
        verdict = judged(os.path.join(d, f"{stem}.judged_v3.2.jsonl"))
        for r in read_jsonl(os.path.join(d, f"{stem}.jsonl")):
            c, j = convs[r["hash"]], verdict.get(r["id"], {})
            msgs = history(c["users"], c["gpt"], r["k"], c.get("system")) + [
                {"role": "assistant", "content": opening[(r["hash"], r["k"])]},
                {"role": "user", "content": r["request"]},
                {"role": "assistant", "content": r["replies"][-1]},
            ]
            runs.append(
                AgentRun(
                    transcripts=[Transcript(messages=messages(msgs))],
                    metadata={
                        "opening_type": kind,
                        "role": c.get("role"),
                        "opening": r["hash"][:40],
                        "turn": r["k"],
                        "projection": r["proj"],
                        "question_id": r["question_id"],
                        "question_category": r["category"],
                        "verdict": j.get("score"),
                        "harmful": j.get("score") in HARMFUL if j else None,
                        "judge_analysis": j.get("analysis"),
                    },
                )
            )
    return runs


def capping_runs():
    sets = load_sets(os.path.join(HERE, "data", "dl"), 25)
    shah = {
        r["id"]: r
        for r in read_jsonl(
            os.path.join(HERE, "data", "shah", "pairs_v3.2.jsonl")
        )
    }
    items = {
        it["id"]: (it["system"], it["turns"])
        for name in ("mhj", "itw")
        for it in sets[name]
    }
    items.update({i: (r["system"], [r["question"]]) for i, r in shah.items()})
    names = {
        "mhj": "MHJ",
        "itw": "DAN",
        "shah": "Shah et al. (regenerated)",
    }
    runs = []
    for name, d in (("mhj", "jb/cap"), ("itw", "jb/cap"), ("shah", "shah/cap")):
        rows = {
            c: {
                r["id"]: r
                for r in read_jsonl(os.path.join(RES, d, f"{name}_{c}.jsonl"))
            }
            for c in ("none", "capped")
        }
        verdict = {
            c: judged(os.path.join(RES, d, f"{name}_{c}.judged_v3.2.jsonl"))
            for c in ("none", "capped")
        }
        for i, r in rows["none"].items():
            system, turns = items[i]
            transcripts, meta = [], {
                "set": names[name],
                "category": r["category"],
            }
            for c, label in (("none", "uncapped"), ("capped", "capped")):
                replies = rows[c][i]["replies"]
                msgs = [{"role": "system", "content": system}] if system else []
                for u, a in zip(turns, replies):
                    msgs += [
                        {"role": "user", "content": u},
                        {"role": "assistant", "content": a},
                    ]
                transcripts.append(
                    Transcript(name=label, messages=messages(msgs))
                )
                meta[f"verdict_{label}"] = verdict[c].get(i, {}).get("score")
            runs.append(AgentRun(transcripts=transcripts, metadata=meta))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        nargs="*",
        choices=list(COLLECTIONS),
        default=list(COLLECTIONS),
    )
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    build = {"drift": drift_runs, "harm": harm_runs, "capping": capping_runs}
    client = (
        Docent(api_key=os.environ["DOCENT_API_KEY"]) if args.upload else None
    )
    for key in args.only:
        runs = build[key]()
        print(f"{key}: converted {len(runs)} agent runs", flush=True)
        print(format_check_report(check_agent_runs(runs)), flush=True)
        if client:
            name, desc = COLLECTIONS[key]
            cid = client.create_collection(name=name, description=desc)
            for s in range(
                0, len(runs), 2000
            ):  # Batches keep each request small.
                client.add_agent_runs(cid, runs[s : s + 2000])
            print(
                f"{key}: uploaded {len(runs)} ->"
                f" https://docent.transluce.org/dashboard/{cid}",
                flush=True,
            )


if __name__ == "__main__":
    main()
