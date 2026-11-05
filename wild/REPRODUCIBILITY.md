# Protected WildChat drift runs

After collecting and classifying conversations with the [selection pipeline](SELECTION.md), run the drift experiment in that selection's output directory:

```bash
python wild/drift_run.py /path/to/new-run
```

This runs generation and extraction in separate Python processes so vLLM workers and GPU memory are released before the Hugging Face extraction model loads. It does not launch or stop a GPU pod, invoke a harm judge, run capping interventions, or upload results. Actual generation requires the project's CUDA/vLLM environment and access to the model and released vectors; the offline tests do not establish that the 70B run fits a particular GPU configuration.

Optional `--model-revision COMMIT_SHA` and `--axis-revision COMMIT_SHA` choose specific Hub versions for the first run. Otherwise the current revisions are resolved once and frozen. The vectors revision pins both `assistant_axis.pt` and `capping_config.pt`; extraction also verifies their file checksums. On resume, omit revision arguments or pass the frozen commit SHAs. Model and tokenizer loaders receive the same pinned model revision.

## What is frozen

`selected.jsonl` must have the `selected.manifest.json` produced by the selection pipeline. Its fingerprint, selected-data checksum, immutable source revision, and retained turn horizon are verified. The whole selection manifest is included in the run identity, including its honest label-review status. The drift runner adopts `selection.max_turns`; it cannot silently measure turns outside the classification window.

`drift.manifest.json` records the selection fingerprints, immutable artifact revisions and vector checksums, generation settings, and measurement convention. Each stage also records fingerprints of its relevant source files and checked Python package versions. Those checks run before cached results are reused. Changed inputs, sidecars, settings, stage code, dependencies, or requested artifact revisions require a new output directory. A manifest prevents accidental mixing of incompatible experiments; it does not guarantee bit-for-bit results across GPU/software platforms or establish that a judge's labels are correct.

The measurement remains one response per turn, temperature 0.7, top-p 0.9, seed 0, BF16, and the original recorded conversation history. Projection still uses the re-encoded response text at layer 40 and the normalized published axis. Multiple samples, sample-specific seeds, and preserving exact generated response token IDs are separate work. Cap firing is measured from the uncapped pass; it is not an intervention result.

## Checkpoints and interruptions

- `drift.state.json` records the stage identity, GPU configuration, planned generated keys, excluded turns, and completion checksum/count. Overlength prefixes and the later turns skipped after them are explicit exclusions. Generation writes `drift.jsonl` atomically only after the batch finishes. Interrupted generation before publication can be rerun, but it restarts the generation batch rather than resuming individual replies.
- `projections.state.json` binds extraction to the completed generation checksum and records its own code/runtime identity. Projection rows are saved incrementally with run/stage identities and record checksums. Resuming validates every saved row and skips only valid completed keys. Empty responses receive an explicit reason with null projection and firing fraction. Finite values, firing-fraction bounds, unique keys, and complete coverage are checked.
- A directory lock prevents two protected generation/extraction processes from writing the same run concurrently. A completed stage also verifies its full output checksum before reuse.

A malformed final projection write or a generation output published without its completion record causes an explicit error. These cases are deliberately not repaired by guessing which measurements finished: preserve the affected directory and use a new run directory containing the unchanged selection files. Existing analysis scripts can still inspect saved data. Outputs without manifests are never silently adopted as protected runs.

## Entry points and legacy experiments

The individual protected stages are `python wild/generate.py RUN_DIR --drift-only` and `python wild/drift_activations.py RUN_DIR`. The drift-only generation command does not require a HarmBench CSV. `drift_run.py` always enters both validators, including when their outputs already exist.

The historical `generate.py RUN_DIR HARMBENCH_CSV` mode and `activations.py` remain available for the role, harm-floor, Shah, and scale-check experiments. They reject writes into a protected drift directory. The combined `run.sh` dispatches extraction to the protected projector when a drift manifest exists, and otherwise retains the legacy projector. Its broader workflow remains outside the new reproducibility guarantees: it still has file-existence shortcuts and pod lifecycle behavior. Use the dedicated drift runner for this experiment. The existing analysis scripts remain unchanged and can read the original output fields plus the new provenance fields.

## Offline validation

```bash
python3 -m pytest wild/tests/test_reproducibility.py -q
```

The tests use small synthetic records and fake model loaders. They exercise the CLI-to-checkpoint flow, unchanged cache reuse without model loading, stale-input/code/settings rejection, corrupted and mixed-run results, generation and extraction interruptions, locks, and propagation of pinned revisions into the actual loader call sites. They do not download datasets or models, contact a GPU service, or produce scientific findings.
