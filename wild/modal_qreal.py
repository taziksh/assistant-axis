"""Harm after drift on real WildChat openings with the paper's own questions
(Qwen 3 32B), on Modal.

Usage (from wild/, MODAL_PROFILE=palaestra; RUN is a folder under results/):
  modal deploy modal_qreal.py
  modal run modal_qreal.py::prepare --run RUN   # upload results/RUN inputs
  modal run modal_qreal.py::launch --run RUN    # start the run; it needs no client
  modal volume get aa-qreal RUN/harm_wild.s0.jsonl results/RUN/
RUN = qreal20: the 20 questions with the highest harm rate on the paper's
openings (2026-11-24). RUN = qreal86: all 86 questions of results/q43b block 1;
its folder starts with qreal20's replies, so only the other 66 are generated.
The Section 4.3 rerun (modal_q43b.py) with real openings in place of the paper's
role openings: the 365 Qwen openings of results/qharm/qfloor (turn 1, most
drifted, spread over bands), each followed by the 20 questions from Lu et al.'s
persona-jailbreak set (see RUN above). Same settings as modal_q43b.py: AXIS_MODEL=qwen
(thinking off), temp 0.7, top-p 0.9, 512 tokens. One GPU; replies are saved to
the Modal Volume aa-qreal after every 20 openings, so a retried container
resumes. The function stops itself after 2 hours at most.
"""

import pathlib
import subprocess

import modal

INPUTS = ("selected.jsonl", "drift.jsonl", "projections.jsonl", "questions.jsonl")
SAVED = "harm_wild.s0.jsonl"  # Replies already generated, if any: the run resumes from them.
SKIP_DIRS = {"results", "data", "figures", "judge_audit", "__pycache__", "tests"}

app = modal.App("aa-qreal")
vol = modal.Volume.from_name("aa-qreal", create_if_missing=True)
hf = modal.Volume.from_name("aa-hf-cache", create_if_missing=True)


def skip(path):
    """True for everything but the Python files outside SKIP_DIRS."""
    return bool(SKIP_DIRS & set(path.parts)) or path.suffix not in ("", ".py")


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.23.0",
        "transformers==4.57.6",
        "accelerate",
        "huggingface_hub",
        "pyarrow",
        "scipy",
        "tqdm",
        "scikit-learn",
        "plotly",
        "openai",
        "python-dotenv",
    )
    .env({"AXIS_MODEL": "qwen", "HF_HOME": "/hf", "PYTHONPATH": "/repo"})
)
if modal.is_local():
    repo = pathlib.Path(__file__).resolve().parent.parent
    image = (
        image.add_local_dir(
            repo / "assistant_axis", "/repo/assistant_axis", ignore=skip
        )
        .add_local_dir(
            repo / "sanity_check" / "scripts",
            "/repo/sanity_check/scripts",
            ignore=skip,
        )
        .add_local_dir(repo / "wild", "/repo/wild", ignore=skip)
    )


@app.function(
    image=image,
    gpu="H200",
    volumes={"/vol": vol, "/hf": hf},
    timeout=2 * 60 * 60,
    retries=2,
)
def shard(run="qreal20"):
    RUN = f"/vol/{run}"
    subprocess.run(
        [
            "python",
            "harm_wild.py",
            RUN,
            "unused_catqa.json",  # Not read with --questions-jsonl.
            "harm",
            "--per-band",
            "100000",
            "--questions-jsonl",
            f"{RUN}/questions.jsonl",
            "--chunk",
            "20",
            "--tag",
            ".s0",
        ],
        cwd="/repo/wild",
        check=True,
    )
    vol.commit()


@app.local_entrypoint()
def prepare(run: str = "qreal20"):
    src = pathlib.Path(__file__).resolve().parent / "results" / run
    with vol.batch_upload(force=True) as batch:
        for name in INPUTS:
            batch.put_file(str(src / name), f"/{run}/{name}")
        if (src / SAVED).exists():
            batch.put_file(str(src / SAVED), f"/{run}/{SAVED}")
    print("inputs uploaded")


@app.local_entrypoint()
def launch(run: str = "qreal20"):
    call = modal.Function.from_name("aa-qreal", "shard").spawn(run)
    print("call", call.object_id)
