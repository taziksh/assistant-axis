"""The Section 4.3 rerun (q43b_build.py, go_q43b.sh) on Modal instead of a pod.

Usage (from wild/):
  modal run modal_q43b.py::prepare   # upload results/q43b inputs, download Qwen
  modal deploy modal_q43b.py
  modal run modal_q43b.py::launch    # start the shards; they run without a client
  modal volume get aa-q43b q43b results/q43b_modal   # fetch the replies
Each shard is one harm_wild.py on its own GPU (--shard I/N), saving after every
5 openings to the Modal Volume aa-q43b, so a retried container resumes. Settings
are go_q43b.sh's: AXIS_MODEL=qwen (thinking off), temp 0.7, top-p 0.9, 512
tokens, the same package versions as sanity_check/scripts/setup.sh. A v1 Volume
allows 5 concurrent writers, hence 5 shards.
"""

import pathlib
import subprocess

import modal

N_SHARDS = 5
RUN = "/vol/q43b"
INPUTS = ("selected.jsonl", "drift.jsonl", "projections.jsonl", "questions.jsonl")
SKIP_DIRS = {"results", "data", "figures", "judge_audit", "__pycache__", "tests"}

app = modal.App("aa-q43b")
vol = modal.Volume.from_name("aa-q43b", create_if_missing=True)
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


@app.function(image=image, volumes={"/hf": hf}, timeout=2 * 60 * 60)
def download():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "Qwen/Qwen3-32B",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )
    hf.commit()


@app.function(
    image=image,
    gpu="H200",
    volumes={"/vol": vol, "/hf": hf},
    timeout=8 * 60 * 60,
    retries=3,
)
def shard(i, n=N_SHARDS):
    subprocess.run(
        [
            "python",
            "harm_wild.py",
            RUN,
            "unused_catqa.json",  # Not read with --questions-jsonl.
            "harm",
            "--per-band",
            "100000",
            "--keep-truncated",
            "--questions-jsonl",
            f"{RUN}/questions.jsonl",
            "--chunk",
            "5",
            "--shard",
            f"{i}/{n}",
            "--tag",
            f".s{i}",
        ],
        cwd="/repo/wild",
        check=True,
    )
    vol.commit()


@app.local_entrypoint()
def prepare():
    src = pathlib.Path(__file__).resolve().parent / "results" / "q43b"
    with vol.batch_upload(force=True) as batch:
        for name in INPUTS:
            batch.put_file(str(src / name), f"/q43b/{name}")
    print("inputs uploaded")
    download.remote()
    print("weights downloaded")


@app.local_entrypoint()
def launch():
    f = modal.Function.from_name("aa-q43b", "shard")
    for i in range(N_SHARDS):
        call = f.spawn(i)
        print("shard", i, call.object_id)
