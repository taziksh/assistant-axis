# History-writer sanity check (Llama 3.3 70B)

Context: "Persona Drift in the Wild" (doc: https://claude.ai/artifact/DEXAmkRTmiNRQHceU39Yrt)
scores turn k of a real WildChat conversation by having the model write reply k after the real
user messages and GPT's earlier replies, then projecting that reply onto the Assistant Axis. This
is the doc's swap test: does it matter that the earlier replies are GPT's, not the model's own?
If not, the method is fine; if so, fall back to real logs (e.g. LMSYS), as in the doc.

Model: Llama 3.3 70B first (released axis at layer 40, capping config). Other models later.

WildChat-50m is not used: it was considered as a source of ready-made Llama replies, but its
generation script cut every message short in long conversations (see below), so most late-turn
replies there were written to a truncated history. Decided 2026-10-31.

Ran 2026-10-31 on 1x B200: 150 conversations, results in
`results/full/` (REPORT.md). Spearman with Llama's own history: GPT 0.90 [0.85, 0.93] vs noise
ceiling 0.96 [0.93, 0.97] vs pirate 0.66; mean offset GPT -0.09 (writing -0.16, coding and
therapy not significant), pirate -0.30. Passes the doc's > 0.8 bar; ranking mostly preserved,
small downward shift.

## Design
- Data: WildChat-1M, English, ≥6 user turns, real user messages and GPT's original replies.
- Unit = (conversation, scored turn k), k in {3, 6, 9} where it exists. About 50 conversations
  each of coding, therapy-like and writing/roleplay (LLM-judge labels, judged by Llama itself).
- Histories for turns 1..k-1:
  - `gpt`: GPT's original replies.
  - `own_a`, `own_b`: two Llama rollouts. Llama answers user message 1, then message 2 after its
    own reply 1, and so on, with no system prompt. Two rollouts give the noise ceiling.
  - `control`: a Llama rollout under a pirate system prompt (first instruction in
    `data/roles/instructions/pirate.json`). Positive control: it must move the projection.
  A rollout stops at the first reply that hits the length limit; later units drop for it.
- Scored reply: Llama writes reply k after each history, no system prompt, 3 samples.
  Temperature 0.7, top_p 0.9 (as in `pipeline/1_generate.py`), up to 1024 new tokens.
- Projection: mean of the layer-40 output over the reply tokens (spans from
  `ConversationEncoder.build_turn_spans`), onto the unit released axis; unit value = mean over
  samples.
- Comparisons: `gpt` vs `own_a` (main), `own_b` vs `own_a` (noise ceiling), `control` vs `own_a`.
- Pass, per domain: Spearman(`gpt`, `own_a`) close to Spearman(`own_b`, `own_a`), small mean
  offset relative to the between-unit SD, and `control` clearly offset. Pooled numbers are
  reported but don't gate. CIs: bootstrap over conversations.
- Known limitation: real user messages were written in reply to GPT, so in the `own` histories
  a user may refer to things Llama never said. No real Llama logs exist to avoid this.

## Verified data facts
- WildChat-1M file 0: 3307 English conversations with ≥6 user turns, none dropped for role
  order, empty content or Llama control strings.
- Released axes: Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B; capping configs only for Qwen and
  Llama (repo README). `llama-3.3-70b/assistant_axis.pt` exists in
  `lu-christina/assistant-axis-vectors`.
- AI-philosophy conversations with ≥6 user turns are rare (about 200 keyword hits across
  WildChat-1M), so that domain is left out.
- WildChat-50m (why it's not used): generated with vllm 0.6.3, temperature 1.0, top_p 1.0,
  max_model_len 8000 for Llama 3.3 (replies capped at 1600 tokens). Every message, GPT replies
  included, was cut to max(256, int(0.95*6000 // n)) tokens, n = message count of the whole
  conversation; prompts over 6000 tokens were chopped; each message carries a stray
  `<|begin_of_text|>`. Share of units with no message cut (WildChat-1M file 0, first row group):
  k=3 64% (174 units), k=6 24% (49), k=9 13% (23).

## Scripts
`scripts/`: build (WildChat-1M) -> judge (domains) -> generate (rollouts, scored turns; vLLM,
tensor parallel 2) -> activations (HF transformers) -> analyze. `run.sh` runs them in order and
skips finished steps; `setup.sh` prepares the pod. Tested locally (2026-10-31): build on one
file; reply spans and prompt prefixes with the Llama 3.1 tokenizer (same chat template); rollout
and scoring logic with a fake vLLM; analyze on synthetic projections. Not tested: real vLLM and
the 70B forward pass. `coverage.py`, `compare_orig.py`, `inspect_shard.py`, `count_wildchat.py`
are WildChat-50m/exploration helpers, unused by `run.sh`.

## Compute
Verified: Llama 3.3 70B bf16 safetensors = 141.1 GB; 80 layers, 8 KV heads, head dim 128, so the
KV cache is 0.33 MB per token (config from the ungated mirror unsloth/Llama-3.3-70B-Instruct).
WildChat-1M sample (60 conversations, ≥6 user turns, Llama tokenizer): GPT reply median 198
tokens (p90 499, none over 1024); prompt at k=3/6/9 median 555/1332/1746 tokens (p90
1536/4082/3977); half the conversations reach turn 9.
Estimate from specs, not measured: on 2x 80GB only about 10 GiB is left for KV cache (about 35k
tokens, so roughly 15-20 conversations at a time). Generation is about 2M tokens in total: roughly
1.5 h on 2x A100, under 1 h on 2x H100, less on 2x H200 (much more KV room). Projections
(about 4500 forward passes through 40 layers, about 1700 tokens each): about 1 h on A100, about
20 min on H100. Total with download and labels: about 3-3.5 h on 2x A100, about 2 h on 2x H100.
The pilot run measures real step times.
The local HF token has no access to meta-llama/Llama-3.3-70B-Instruct, so the scripts use the
ungated mirror unsloth/Llama-3.3-70B-Instruct: all 30 safetensors match Meta's sha256 and the
chat template matches Meta's Llama 3.1 template (checked 2026-10-31); only pad settings differ.
Pod: 2x A100 SXM (`NVIDIA A100-SXM4-80GB`). Run in tmux (`NAME=pilot
PER_DOMAIN=4 STOP_POD=0 bash run.sh` first); the pod stops itself on exit or after the hour cap.
Local runpodctl (2.7.2) is logged in to the user's own account, which has no balance (2026-10-31).
