"""Shared settings for the drift / harm / capping experiments on real chats
(Llama 3.3 70B).
"""

import importlib.util
import os
from wild.common.paths import REPO_ROOT

# Load the sanity check's common.py by path: putting its folder on sys.path
# would let its other
# scripts (e.g. coverage.py) shadow real packages that vLLM imports.
_spec = importlib.util.spec_from_file_location(
    "sanity_common",
    REPO_ROOT / "sanity_check" / "scripts" / "common.py",
)
_sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sc)
AXIS_FILE, AXIS_REPO, BAD_STRINGS, LAYER, MODEL = (
    _sc.AXIS_FILE,
    _sc.AXIS_REPO,
    _sc.BAD_STRINGS,
    _sc.LAYER,
    _sc.MODEL,
)
TEMPERATURE, TOP_P = _sc.TEMPERATURE, _sc.TOP_P
history, prompt_ids, read_jsonl, write_jsonl = (
    _sc.history,
    _sc.prompt_ids,
    _sc.read_jsonl,
    _sc.write_jsonl,
)

DOMAINS = [
    "coding",
    "writing",
    "therapy",
    "ai",
]  # The paper's four drift domains.
# Score user turns 1..12 (the paper's simulated conversations ran about this
# long).
MAX_TURN = 12
MAX_MODEL_LEN = 16384
MAX_NEW_TOKENS = 1024
MAX_PROMPT_TOKENS = MAX_MODEL_LEN - MAX_NEW_TOKENS
HARM_NEW_TOKENS = 512  # HarmBench's generation length.

CAPPING_FILE = "llama-3.3-70b/capping_config.pt"
CAPPING_EXPERIMENT = (  # Recommended setting for Llama 3.3 70B (repo README).
    "layers_56:72-p0.25"
)
HARM_CLS = "cais/HarmBench-Llama-2-13b-cls"

# The paper says only "deepseek-v3" (no version, provider or date; checked the
# arXiv text and the paper
# source in the repo history). Main judge (user's call, 2026-11-02): the latest
# V3 before the paper
# (posted 2026-01-15), V3.2 (2025-12-01), which DeepSeek's own API served as
# "deepseek-chat" (thinking
# off) at that time. DeepSeek's API no longer serves it, so we pin open weights
# on OpenRouter: one model
# id, one provider endpoint, no fallbacks, thinking off. Every reply logs the
# model and provider used.
# Endpoint tags from openrouter.ai/api/v1/models/<id>/endpoints (checked
# 2026-11-03).
JUDGES = {
    "v3.2": {
        "model": "deepseek/deepseek-v3.2",
        "endpoint": "siliconflow/fp8",
        "provider": "SiliconFlow",
    },  # Native fp8.
    "v3": {
        "model": "deepseek/deepseek-chat",
        "endpoint": "deepinfra/fp4",
        "provider": "DeepInfra",
    },  # Original V3, 2024-12-26.
    "v3-0324": {
        "model": "deepseek/deepseek-chat-v3-0324",
        "endpoint": "siliconflow/fp8",
        "provider": "SiliconFlow",
    },
}


def unit_axis():
    """The released Assistant Axis at LAYER, unit length (float32)."""
    from huggingface_hub import hf_hub_download
    import torch.nn.functional as F

    from assistant_axis import load_axis

    return F.normalize(
        load_axis(hf_hub_download(AXIS_REPO, AXIS_FILE, repo_type="dataset"))[
            LAYER
        ].float(),
        dim=0,
    )


def capping_config():
    """The released capping config (use with CAPPING_EXPERIMENT)."""
    from huggingface_hub import hf_hub_download

    from assistant_axis import load_capping_config

    return load_capping_config(
        hf_hub_download(AXIS_REPO, CAPPING_FILE, repo_type="dataset")
    )


def load_probing_model():
    """Llama 3.3 70B in HF. cuDNN's attention kernel crashed on large padded
    batches (2026-11-02:

    mha_graph.execute ... is_good()), so it is turned off for every HF script.
    """
    import torch

    from assistant_axis.internals import ProbingModel

    torch.backends.cuda.enable_cudnn_sdp(False)
    return ProbingModel(MODEL)


EMBED_MODEL = (  # The paper's Section 4.2 embedding model.
    "Qwen/Qwen3-Embedding-0.6B"
)


def harmbench_standard(path):
    """HarmBench standard text behaviors (no contextual or copyright ones), as
    dicts.
    """
    import csv

    with open(path) as f:
        return [
            r
            for r in csv.DictReader(f)
            if r["FunctionalCategory"] == "standard"
        ]
