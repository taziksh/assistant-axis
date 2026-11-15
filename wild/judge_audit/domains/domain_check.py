"""Blind hand-check of the WildChat domain classifier (therapy / AI philosophy).

The Docent collection "AA persona drift: domain spot-check" holds real WildChat
chats that a Llama 3.3 70B classifier labelled therapy or ai_philosophy. This
builds a page where a reviewer labels each chat's domain without seeing the
classifier's label, then scores precision.

Usage:
  domain_check.py build COLLECTION_ID OUT_DIR [--seed 0]
      Fetches the collection to OUT_DIR/docent_runs.jsonl (reused if present;
      needs DOCENT_API_KEY in the environment or the repo's .env) and writes
      OUT_DIR/items.jsonl and OUT_DIR/domaincheck.html.
  domain_check.py score LABELS.jsonl [--items OUT_DIR/items.jsonl]
      Precision of the classifier against exported labels: overall, by domain,
      by pool, by subtopic, with Wilson 95% intervals.

Reviewer labels: therapy, ai_philosophy, neither, unsure. A chat counts for a
domain only if the user's messages across the whole chat are about it;
roleplay, chats that switch to an unrelated task, and plain factual questions
do not count (the collection's own instructions).
"""

import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DOMAINS = ("therapy", "ai_philosophy")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def load_env():
    for d in (REPO, os.path.dirname(REPO)):
        p = os.path.join(d, ".env")
        if os.path.exists(p):
            for line in open(p):
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def text_of(content):
    if isinstance(content, str):
        return content
    return "".join(c.get("text", "") for c in content or [] if isinstance(c, dict))


def fetch(collection_id, cache):
    load_env()
    from docent import Docent
    c = Docent(api_key=os.environ["DOCENT_API_KEY"])
    runs = []
    for rid in c.list_agent_run_ids(collection_id):
        run = c.get_agent_run(collection_id, rid)
        if run is None:
            continue
        run = run.model_dump(mode="json") if hasattr(run, "model_dump") else run
        ts = run.get("transcripts") or []
        ts = list(ts.values()) if isinstance(ts, dict) else ts
        msgs = [{"role": m["role"], "text": text_of(m.get("content"))}
                for t in ts for m in (t.get("messages") or [])]
        runs.append({"id": run["id"], "metadata": run.get("metadata") or {}, "messages": msgs})
    write_jsonl(cache, runs)
    print(f"fetched {len(runs)} runs from {collection_id} -> {cache}", file=sys.stderr)
    return runs


def build(args):
    cache = os.path.join(args.out_dir, "docent_runs.jsonl")
    runs = read_jsonl(cache) if os.path.exists(cache) else fetch(args.collection_id, cache)
    items = []
    for r in runs:
        m = r["metadata"]
        items.append({
            "key": m.get("hash") or r["id"], "run_id": r["id"],
            "domain": m.get("classifier_domain"), "subtopic": m.get("classifier_subtopic"),
            "pool": m.get("pool"), "turns": m.get("turns"),
            "messages": r["messages"],
        })
    bad = [i["key"] for i in items if i["domain"] not in DOMAINS]
    if bad:
        sys.exit(f"unexpected classifier_domain on {len(bad)} runs, e.g. {bad[:3]}")
    items.sort(key=lambda i: i["key"])
    random.Random(args.seed).shuffle(items)  # Mixes domains and pools; fixed so pages match across rebuilds.
    payload_hash = hashlib.sha1(json.dumps([i["key"] for i in items]).encode()).hexdigest()[:12]
    write_jsonl(os.path.join(args.out_dir, "items.jsonl"), items)
    meta = {"collection_id": args.collection_id, "payload_hash": payload_hash, "seed": args.seed,
            "n": len(items)}
    meta["counts"] = {f"{d}/{p}": n for (d, p), n in collections.Counter((i["domain"], i["pool"]) for i in items).items()}
    payload = json.dumps({"meta": meta, "items": items}).replace("</", "<\\/")
    page = open(os.path.join(HERE, "domaincheck_page.html")).read()
    assert page.count("/*__PAYLOAD__*/null") == 1
    out = os.path.join(args.out_dir, "domaincheck.html")
    with open(out, "w") as f:
        f.write(page.replace("/*__PAYLOAD__*/null", payload))
    print(f"wrote {out} ({len(items)} chats, payload {payload_hash}); counts {meta['counts']}")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def score(args):
    items = {i["key"]: i for i in read_jsonl(args.items)}
    labels = {r["key"]: r for r in read_jsonl(args.labels) if r.get("label")}
    rows = [(items[k], labels[k]) for k in labels if k in items]
    missing = len(items) - len(rows)
    print(f"{len(rows)} labeled chats of {len(items)}" + (f" ({missing} unlabeled)" if missing else ""))
    print("Precision = share of the classifier's picks you labeled with the same domain; "
          "'unsure' is left out of precision and counted separately.\n")

    def line(name, group):
        sure = [(it, L) for it, L in group if L["label"] != "unsure"]
        k = sum(L["label"] == it["domain"] for it, L in sure)
        p, lo, hi = wilson(k, len(sure))
        uns = len(group) - len(sure)
        print(f"| {name} | {len(group)} | {k}/{len(sure)} | {100 * p:.0f}% ({100 * lo:.0f}, {100 * hi:.0f}) | {uns} |")

    print("| Group | n | Correct | Precision (95% CI) | Unsure |\n|---|---|---|---|---|")
    line("all", rows)
    for d in DOMAINS:
        line(d, [x for x in rows if x[0]["domain"] == d])
    for d in DOMAINS:
        for p in sorted({it["pool"] for it, _ in rows}):
            g = [x for x in rows if x[0]["domain"] == d and x[0]["pool"] == p]
            if g:
                line(f"{d}, pool {p}", g)
    print("\nYour label by classifier domain:")
    conf = collections.Counter((it["domain"], L["label"]) for it, L in rows)
    labs = ("therapy", "ai_philosophy", "neither", "unsure")
    print("| Classifier \\ You | " + " | ".join(labs) + " |\n|---|" + "---|" * len(labs))
    for d in DOMAINS:
        print(f"| {d} | " + " | ".join(str(conf[(d, l)]) for l in labs) + " |")
    print("\nBy subtopic (correct / labeled excluding unsure):")
    for d in DOMAINS:
        sub = collections.defaultdict(list)
        for it, L in rows:
            if it["domain"] == d and L["label"] != "unsure":
                sub[it["subtopic"]].append(L["label"] == d)
        print(f"  {d}: " + ", ".join(f"{s} {sum(v)}/{len(v)}" for s, v in sorted(sub.items(), key=lambda x: -len(x[1]))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("collection_id")
    b.add_argument("out_dir")
    b.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("labels")
    s.add_argument("--items", default=os.path.join(REPO, "wild", "results_docent_domains", "items.jsonl"))
    args = ap.parse_args()
    build(args) if args.cmd == "build" else score(args)


if __name__ == "__main__":
    main()
