"""Shared constants and helpers for the history-writer sanity check."""
import json

# Ungated mirror of meta-llama/Llama-3.3-70B-Instruct: same safetensors (sha256) and chat template.
MODEL = "unsloth/Llama-3.3-70B-Instruct"
LAYER = 40
AXIS_REPO, AXIS_FILE = "lu-christina/assistant-axis-vectors", "llama-3.3-70b/assistant_axis.pt"

# Histories for the scored turn: gpt is WildChat-1M's original; own_a/own_b are two Llama rollouts
# on its own earlier replies (noise ceiling); control is a Llama rollout under CONTROL_SYSTEM; none is
# no history at all (only user message k).
HISTORY_WRITERS = ["gpt", "own_a", "own_b", "control", "none"]
DOMAINS = ["coding", "therapy", "writing"]
SCORED_TURNS = (3, 6, 9)

# Temperature and top_p match pipeline/1_generate.py. MAX_NEW_TOKENS is above its 512 so that fewer
# rollout replies hit the limit (that makes the rest of the rollout unusable).
TEMPERATURE, TOP_P, MAX_NEW_TOKENS = 0.7, 0.9, 1024
SCORED_SAMPLES = 3
MAX_MODEL_LEN = 16384
MAX_PROMPT_TOKENS = MAX_MODEL_LEN - MAX_NEW_TOKENS

# First pirate instruction in data/roles/instructions/pirate.json.
CONTROL_SYSTEM = "You are a pirate captain who has sailed the seven seas in search of treasure and adventure."

# Text that the Llama tokenizer would turn into control tokens and break span finding.
BAD_STRINGS = ("<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
               "<|eom_id|>", "<|python_tag|>", "<|finetune_right_pad_id|>")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def history(users, replies, k, system=None):
    """Messages through user turn k (1-based), with the writer's replies to turns 1..k-1."""
    msgs = [{"role": "system", "content": system}] if system else []
    for j in range(k - 1):
        msgs += [{"role": "user", "content": users[j]}, {"role": "assistant", "content": replies[j]}]
    return msgs + [{"role": "user", "content": users[k - 1]}]


def context(c, k, writer):
    """Messages the scored reply to user turn k is generated after, for one history writer."""
    if writer == "none":
        return [{"role": "user", "content": c["users"][k - 1]}]
    return history(c["users"], c["replies"][writer], k)


def usable(unusable, k):
    """A unit (conversation, k) is kept for a writer if all its replies before turn k are usable."""
    return not any(unusable[: k - 1])


def scored_turns(n_users):
    return [k for k in SCORED_TURNS if k <= n_users]


def prompt_ids(tok, msgs):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids
