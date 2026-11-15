"""Prepare Lu et al.'s persona jailbreak data for our judge and capping runs.

Usage: lu_prep.py
Reads data/lu/ (local, git-ignored; see its PROVENANCE.md) and writes:
- data/lu/pairs.jsonl: system prompt, question, persona and category per pair,
  in the format of capping.py --shah.
- results/lu/baseline_{llama,qwen}.jsonl: their saved uncapped replies and
  verdicts, in the format of judge_paper.py, to check our judge against theirs.
The category in EXCLUDED is left out of everything.
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "lu")
OUT = os.path.join(HERE, "results", "lu")
EXCLUDED = {"Endorsing pedophilia"}
MODELS = {"llama": "llama-3.3-70b", "qwen": "qwen-3-32b"}


def write(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(DATA, "jailbreak_1100.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    kept = [r for r in rows if r["harm_category"] not in EXCLUDED]
    pairs = {
        r["question"]: {
            "id": f"lu{r['id']}",
            # {company} is filled in for generation, as in their runs. The judge's
            # request is the raw system prompt plus the question: of the mappings
            # their code allows, it agrees best with their saved labels
            # (92.5% vs 89.3% on Llama, 93.5% vs 90.5% on Qwen).
            "system": r["prompt"].format(company="Meta"),
            "judge_request": r["prompt"] + " " + r["question"],
            "question": r["question"],
            "persona": r["persona"],
            "category": r["harm_category"],
        }
        for r in kept
    }
    write(os.path.join(DATA, "pairs.jsonl"), list(pairs.values()))
    print(f"pairs: {len(pairs)} of {len(rows)}", flush=True)
    # Qwen: their command fills {company}=Alibaba, {name}=Qwen.
    qwen = {r["question"]: r["prompt"].format(company="Alibaba", name="Qwen") for r in kept}
    write(
        os.path.join(DATA, "pairs_qwen.jsonl"),
        [{**p, "system": qwen[q]} for q, p in pairs.items()],
    )
    # Their own script (capped/1_capping_vllm.py) reads this format directly.
    write(os.path.join(DATA, "theirs_1075.jsonl"), kept)
    for short, model in MODELS.items():
        out = []
        for path in glob.glob(os.path.join(DATA, "viewer", model, "c*.json")):
            with open(path) as f:
                chunk = json.load(f)
            for r in chunk:
                p = pairs.get(r["question"])
                if p is None:
                    continue
                saved = r["responses"]["contrast"]["0"]["jailbreak"]
                out.append({
                    "id": p["id"],
                    "persona": p["persona"],
                    "category": p["category"],
                    "request": p["judge_request"],
                    "replies": [saved["response"]],
                    "their_score": saved["score"],
                })
        write(os.path.join(OUT, f"baseline_{short}.jsonl"), out)
        print(f"baseline_{short}: {len(out)}", flush=True)


if __name__ == "__main__":
    main()
