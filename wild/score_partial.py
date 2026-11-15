"""Project the replies of MHJ attacks that are already finished in the uncapped run's checkpoint.

Usage: score_partial.py RUN_DIR DL_DIR
For an early look at the harm test while the uncapped MHJ run is paused between turns. Same projection as
capping.py (project_replies). Writes RUN_DIR/cap/mhj_none_proj.partial.jsonl; capping.py later projects
all attacks into mhj_none_proj.jsonl.
"""
import argparse, json, os
from tqdm import tqdm
from assistant_axis.internals import ProbingModel
from capping import project_replies  # Also turns off cuDNN attention.
from jbsets import load_sets
from wcommon import MODEL, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("dl_dir")
    args = ap.parse_args()
    items = {it["id"]: it for it in load_sets(args.dl_dir, 25)["mhj"]}
    ck = json.load(open(os.path.join(args.run_dir, "cap", "mhj_none.jsonl.ckpt")))
    done = [items[i] for i, r in ck.items() if len(r) == len(items[i]["turns"])]
    print(f"projecting {len(done)} finished attacks", flush=True)
    pm = ProbingModel(MODEL)
    write_jsonl(os.path.join(args.run_dir, "cap", "mhj_none_proj.partial.jsonl"),
                [{"id": it["id"], "proj": project_replies(pm, it, ck[it["id"]])} for it in tqdm(done, mininterval=30)])


if __name__ == "__main__":
    main()
