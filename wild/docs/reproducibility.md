# Protected WildChat drift runs

After collecting and classifying conversations with the [selection pipeline](selection.md), run the drift experiment in that selection's output directory:

```bash
python -m wild.drift /path/to/new-run
```

This runs generation, extraction, and CPU analysis in separate Python processes so vLLM workers and GPU memory are released before the Hugging Face extraction model loads. Actual generation requires the project's CUDA/vLLM environment and access to the model and released vectors; the offline tests do not establish that the 70B run fits a particular GPU configuration. See [the drift pilot guide](drift.md) for the selection and plot workflow.

Optional `--model-revision COMMIT_SHA` and `--axis-revision COMMIT_SHA` choose specific Hub versions for the first run. Otherwise the current revisions are resolved once and frozen. The vectors revision pins both `assistant_axis.pt` and `capping_config.pt`; extraction also verifies their file checksums. On resume, omit revision arguments or pass the frozen commit SHAs. Model and tokenizer loaders receive the same pinned model revision.

## What is frozen

`selected.jsonl` must have the `selected.manifest.json` produced by the selection pipeline. Its fingerprint, selected-data checksum, immutable source revision, and retained turn horizon are verified. The whole selection manifest is included in the run identity, including its honest label-review status. The drift runner adopts `selection.max_turns`; it cannot silently measure turns outside the classification window.

`drift.manifest.json` records the selection fingerprints, immutable artifact revisions and vector checksums, generation settings, and measurement convention. Each stage also records fingerprints of its relevant source files and checked Python package versions. Those checks run before cached results are reused. Changed inputs, sidecars, settings, stage code, dependencies, or requested artifact revisions require a new output directory. A manifest prevents accidental mixing of incompatible experiments; it does not guarantee bit-for-bit results across GPU/software platforms or establish that a judge's labels are correct.

New runs generate three independent replies per turn. `--samples-per-turn N` changes that count; omitting it on resume uses the saved count. The model, layer, BF16 precision, temperature 0.7, top-p 0.9, and original recorded history stay fixed. Every reply has a seed derived from base seed zero, its conversation ID, turn, and sample index. Reordering inputs does not change these seeds. Numerical behavior can still differ across GPU/software platforms or scheduling conditions.

Generation saves the actual prompt token IDs and generated response token IDs returned by vLLM. Projection uses those IDs directly, averaging layer-40 post-block residual activations over generated content tokens and projecting onto the normalized published axis. Tokenizer special IDs, EOS IDs from the pinned model's generation configuration, and an integer completion stop reason are excluded from the average. Special tokens remain in the forward context when they precede a measured token. Text is saved for inspection and is not re-encoded for measurement. Cap firing is measured from the uncapped pass; it is not an intervention result.

The sample/token format uses manifest schema version 2. Earlier protected runs used one reply and re-encoded text; new commands reject those manifests rather than combining measurement conventions. Preserve old directories and create a fresh run with the unchanged selection files when changing the protocol.

## Checkpoints and interruptions

- `drift.state.json` records the stage identity, GPU configuration, planned conversation/turn/sample keys, prompt fingerprints, token exclusions, and completion checksum/count. Overlength prefixes and the later turns skipped after them exclude every sample of the affected turn explicitly. Generation works in fixed batches of at most 16 requests and saves each returned reply with a record checksum before advancing. After interruption, it validates and skips saved samples; only unsaved replies are regenerated. Completion is published after all planned samples are present.
- `projections.state.json` binds extraction to the completed generation checksum and records its own code/runtime identity. Projection rows are saved incrementally with run/stage identities and record checksums. Resuming validates every saved row and skips only valid completed sample keys. Replies with no scored content tokens receive `reason: "no_content_tokens"`, a zero content-token count, and null projection/firing fraction. Finite values, token counts, truncation flags, firing-fraction bounds, unique keys, and complete coverage are checked.
- A directory lock prevents two protected generation/extraction processes from writing the same run concurrently. A completed stage also verifies its full output checksum before reuse.

A malformed final generation or projection write causes an explicit error. These cases are not repaired by guessing which measurements finished: preserve the affected directory and use a new run directory containing the unchanged selection files. A fully written set of valid samples can be finalized on resume if only its completion publication was interrupted. Outputs without manifests are never silently adopted as protected runs.

## Entry points and legacy experiments

The individual protected stages are `python -m wild.drift.generation RUN_DIR`, `python -m wild.drift.measurement RUN_DIR`, and `python -m wild.drift.analysis RUN_DIR`. The runner always enters the validators, including when outputs already exist. Completed result files can be copied to a Mac for analysis: the CPU reader checks saved identities, input/output checksums, sample coverage, and stage links without requiring the GPU machine's installed packages or current source files.

Historical single-reply generation and text-based measurement live in `wild.legacy.generation` and `wild.legacy.measurement`. The combined `wild/scripts/run.sh` workflow dispatches protected runs to the new modules, resumes partial generation, and defers analysis until projection extraction is complete. Its broader harm/capping workflow retains its existing cache shortcuts and pod lifecycle behavior. Use `python -m wild.drift` for the drift pilot. `wild.reports.analyze` routes schema-v2 drift runs to the new analysis so repeated samples cannot overwrite each other.

Use `python -m wild COMMAND ...` to retain an old command name and arguments, such as `python -m wild generate RUN_DIR --drift-only`. The old root-level `.py` paths have been removed. See the [directory guide](../README.md) for the package layout. Source-file moves invalidate execution fingerprints on interrupted protected runs; resume those with their original checkout. Completed result analysis does not require the original source layout.

## Offline validation

```bash
python3 -m pytest wild/tests/test_reproducibility.py wild/tests/test_reply_sampling.py -q
```

The tests use small synthetic records and fake model loaders. They exercise independent seeds, actual completion token preservation, unchanged cache reuse, stale-input/code/settings rejection, corrupted and mixed-run results, generation and extraction interruptions, locks, and propagation of pinned revisions into the actual loader call sites. When PyTorch and Transformers are installed, a tiny randomly initialized Llama test compares the hook result with a direct layer-activation calculation, including exclusion of special tokens inside the response. No tests download datasets or model weights, contact a GPU service, or produce scientific findings.
