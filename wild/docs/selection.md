# Four-domain WildChat selection

`wild.selection.build` collects a versioned candidate pool and `wild.selection.judge` selects coding, writing, emotional-disclosure (`therapy`), and AI-philosophy (`ai`) conversations. Their outputs retain the existing `hash`, `users`, `gpt`, and `domain` fields consumed by the drift scripts. This stage does not generate experimental replies or measure projections.

## Selection protocol

The default eligibility threshold is three complete exchanges, configurable with `--min-turns`. The retained window is the initial twelve exchanges or the conversation's end, configurable with `--max-turns`. Original text and earlier history are preserved. Changing these values changes the evaluated population; use a new output location and pass the same window settings to both commands. Later analyses must report the number of conversations contributing at each turn.

The builder reads English conversations with clean user/assistant alternation from WildChat-4.8M. It resolves the requested dataset version to an immutable Hub commit before listing or reading files. The default candidate pool includes therapy/AI keyword matches and up to 8,000 other eligible conversations. `--random N` instead samples uniformly from the eligible, deduplicated pool. Both modes use `--seed` (default zero). Duplicate conversation IDs, identical transcripts, full shorter snapshots, and shared source-turn identifiers are removed deterministically, retaining longer snapshots first. A common greeting alone does not establish duplication. There is no fuzzy semantic deduplication or network-identity filter.

The judge sees every user message in the retained window, without character clipping or assistant responses. It classifies the user topic, using the versioned policy in `wild/selection/helpers.py`, and returns a category plus a short rationale. Coding covers programming and debugging; writing covers drafting/editing, including fiction as a writing task; therapy covers personal emotional disclosure and support; AI philosophy covers the AI's consciousness, feelings, identity, agency, or nature. Substantive mixed topics, unrelated/non-English content, and attempts to make the assistant inhabit a persona, roleplay a relationship, or override its rules are excluded. The persona/roleplay exclusion applies across all four categories. Ordinary greetings and clarifying follow-ups do not make a conversation mixed.

A window that cannot fit the judge's context budget is explicitly marked `too_long`; malformed judge output is marked `unparsed`. Neither is silently shortened or assigned a fallback domain. These exclusions and the pure-domain policy narrow the population. Keyword enrichment, LLM classification, and length exclusions produce a targeted exploratory sample, not a representative estimate over all WildChat.

## Commands

From the repository root, prepare a fresh candidate location and selection directory:

```bash
python -m wild.selection.build wild/data/convs_v1.jsonl --min-turns 3 --max-turns 12
python -m wild.selection.judge wild/data/convs_v1.jsonl wild/results/drift_v1 --per-domain 10 --min-turns 3 --max-turns 12
```

The builder downloads dataset records. The judge loads Llama 3.3 70B and requires the existing CUDA/vLLM environment. These commands are examples for a future experiment; the code changes were validated with offline tests only. Supply `--revision COMMIT_SHA` to the builder and `--model-revision COMMIT_SHA` to the judge to choose exact versions yourself. The default `main` is resolved once and the saved exact revision is reused on an identical rerun. Judge settings include a 16,384-token context budget, a 256-token output allowance, greedy decoding, and a fixed seed.

## Review and provenance

The candidate file has a sibling `<filename>.manifest.json` containing the dataset commit, source paths, collection settings, code hashes, filter counts, and output checksums. `<filename>.duplicates.jsonl` records discarded overlaps and their retained counterparts.

The selection directory contains:

- `selected.jsonl`: the selected conversations and assigned domains.
- `labels.jsonl`: each candidate's classification, rationale, raw judge response, and prompt length, including exclusions.
- `selection.audit.jsonl`: the selection outcome for every candidate, including quota exclusions.
- `selected.manifest.json`: exact judge revision, policy/configuration fingerprints, candidate provenance, output checksums, per-category counts and shortages, and `review_status: "unreviewed"`.

Inspect the labels and rationales before using domain comparisons. Machine-generated rationales are not human validation. This increment does not implement label adjudication or silently mark labels approved. Preserve review notes separately; a label-policy or data correction should produce a fresh selection rather than editing a frozen run's files. The drift-run safeguards consume the entire selection manifest, including the retained turn horizon at `selection.max_turns`.

Outputs are atomically replaced and the completion manifest is published last. Identical reruns verify every output checksum before skipping work. A fingerprint of the whole manifest detects accidental changes to its provenance, counts, and review status. Changed inputs, policy, code, judge settings or runtime versions, missing outputs, modified files, and legacy files without manifests cause an explicit error. Use a new output location for a changed protocol or an interrupted publication. Nothing overwrites or retroactively certifies old runs. A local lock prevents two selection processes from writing the same outputs concurrently.

The old 40-conversation pilot remains separate: it uses WildChat-1M, different length criteria, and an agent-reviewed selection protocol. It should not be pooled with new WildChat-4.8M selections without an explicit study design.

## Offline checks

```bash
python3 -m pytest wild/tests/test_selection.py -q
```

These tests exercise full-window handling, preserved text, overlap checks, strict label parsing, deterministic quotas/shortfalls, immutable source paths, cache rejection, interrupted publication, and a mocked selection run. They do not establish real judge accuracy or GPU fit. Multiple experimental replies per turn and new four-category figures remain subsequent work.
