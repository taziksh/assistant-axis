# Prepared commands for the next drift run

Use these commands in a later user-run pod session. Preserve all earlier candidate, label, smoke, and raw measurement files. The new defaults are `wild/data/drift_candidates_v3.jsonl`, `wild/results/drift_smoke_v3`, and `wild/results/drift_v3`.

The prepared pod checkout is `/workspace/palaestra`, with its existing `.venv` and model cache at `/workspace/palaestra/hf`. `wild/scripts/pilot.sh` pins `unsloth/Llama-3.3-70B-Instruct` to `99cd0d2c829e92a67c844f9144c2509632e5c87f`, reuses that cache, and requires the pinned snapshot to exist. It sets `HF_HUB_ENABLE_HF_TRANSFER=0`, overriding the pod's inherited value of `1`; the optional `hf_transfer` package is absent. This avoids the earlier transfer-package failure without installing it or redownloading weights.

The previous successful smoke only verifies its old code and candidate fingerprints. The classifier, collection protocol, and other relevant code have changed, so it cannot authorize the next run. A fresh smoke under the final code is required. Keep diagnostics separate from scientific measurements.

## Prepare locally before any GPU stage

Review the update and run local tests. The revised target is 100 conversations in each of four domains, up to fifteen original exchanges, with three sampled replies at each turn. The paper uses that conversation count and horizon, but the count is not a power calculation and three replies are not independent conversations.

The existing 8,284-candidate pool contains only the first twelve exchanges from three source files. It cannot support an honest fifteen-turn experiment by changing a flag. Prepare a new candidate pool from the original pinned source, following the bounded twelve-file example in [selection.md](selection.md). That command uses a reproducible subset across the full pinned listing, small Arrow batches, at most two workers, and up to 8,000 non-keyword candidates. It has not been run as part of these changes.

The separate local validation-source collection uses ten fresh pinned files and preserves original windows up to fifteen turns. Its results and exact coverage are recorded in `wild/results/selection_validation_v3/SOURCE_REPORT.md`. That reference-building source artifact is separate from the production candidate pool expected by `pilot.sh`; it does not claim a selected 400-conversation study cohort.

After collection, inspect the manifest's source paths, coverage, turn window, filter/deduplication counts, and retrieval distribution. Fresh validation references use original windows up to fifteen turns, with deliberate searches for genuine therapy and AI positives and actual turns 13–15. Check their achieved coverage and unresolved shortages in `wild/results/selection_validation_v3/readiness.json`; a fifteen-turn setting alone does not establish long-window coverage. The older audited references cover only their original twelve-turn windows. Do not select only long conversations or extreme emotional/philosophical examples to obtain a full-looking late-turn curve.

When ready for a future pod session, transfer the reviewed code, fresh candidate JSONL and both sidecars, and local validation references with their manifest, exact source artifact, and exclusion artifact at the relative paths recorded in that manifest. An older update archive is stale after these changes; build and inspect a new archive before transfer. No archive or upload is assumed complete here. The helper installs nothing and manages no pod lifecycle.

## 1. Run a fresh GPU smoke

From your own future SSH session in `/workspace/palaestra`:

```bash
bash wild/scripts/pilot.sh smoke
```

The diagnostic exercises the production classifier, three sampled replies for each of two original-history prefixes, exact-token projection, and a synthetic 16,384-token projection stress check. It writes to `wild/results/drift_smoke_v3`. Success requires a zero exit code and a verified `status: "complete"` result. Inspect the output and log. It verifies this small workload, not label accuracy, memory use for every possible input, or a scientific result.

After changing code or correcting a failure, use another fresh directory and retain it for later stages in that terminal:

```bash
export SMOKE_DIR=wild/results/drift_smoke_v4
bash wild/scripts/pilot.sh smoke
```

## 2. Validate the revised classifier on the small reference set

```bash
bash wild/scripts/pilot.sh validate
```

The defaults are `CLASSIFIER_REFERENCES=wild/results/selection_validation_v3/references.jsonl` and `VALIDATION_DIR=wild/results/selection_validation_v3/model_run`. Before loading the model, evaluation verifies the frozen policy, taxonomy, full original windows, exact source records, and separation from previously screened conversations. It uses the pinned production judge and saves predictions, disagreements, domain/subtopic agreement, coverage, and a sealed manifest. Inspect invalid/truncated outputs and every disagreement, keeping any regression and held-out strata separate. References are provisional AI review, not human gold labels; no accuracy threshold or automatic scientific approval is inferred from a completed command. The evaluator can run a diagnostic on an incomplete reference set, but its coverage report must retain the shortages. Code tests, reference coverage, and actual classifier results are separate milestones.

If a policy or reference correction is needed, preserve the earlier evaluation and choose fresh reference/output paths. Do not edit a completed evaluation's inputs in place. The stage stops after evaluation.

## 3. Select and review the target 400 conversations

```bash
bash wild/scripts/pilot.sh select
less wild/results/drift_v3/selection.review.txt
```

Selection classifies the full new candidate pool and targets 100 conversations per domain, retaining three to fifteen original exchanges. It balances named subtopics as evenly as availability permits, redistributes scarce buckets' places within the domain, and then uses valid broad/unassigned-subtopic conversations if needed. It records the frozen quota, achieved balance, fallback counts, and horizon in `selected.manifest.json`, generates a review file with each selected label, rationale, and full transcript, then stops. Inspect category fit, live language, mixed tasks, and fictional identity/relationship or rule-override framing. Ordinary professional task expertise is allowed. The focused therapy and AI definitions are in [selection.md](selection.md).

Inspect all selected labels, `labels.jsonl`, `selection.audit.jsonl`, and per-category shortages before measurement. The manifest remains `review_status: "unreviewed"`; the helper does not claim a review occurred. Preserve review notes separately. A label-policy or source correction requires a fresh selection. The old machine counts included only 83 therapy candidates before review, so the larger pool is needed, and even it may fall short.

`PER_DOMAIN` is a real optional quota setting, useful for a deliberately smaller diagnostic cohort. Use a fresh run directory when changing it, for example:

```bash
export RUN_DIR=wild/results/drift_small_diagnostic_v3
export PER_DOMAIN=10
bash wild/scripts/pilot.sh select
```

That is a ten-per-domain diagnostic, not completion of the 100-per-domain target. Measurement checks categories against the quota frozen in the selection manifest, so later environment changes cannot silently redefine the cohort. Do not claim a full-size run if the frozen quota is smaller.

## 4. Measure only after reviewing selection

```bash
bash wild/scripts/pilot.sh measure
```

The helper verifies the successful smoke against its candidate pool, checks that selection used the same pool/provenance, and requires every domain to meet the frozen quota. It then generates three replies per eligible turn using recorded history, measures exact-token projections, and writes `analysis/REPORT.md` plus figures. Model and axis revisions come from the verified smoke. The turn horizon comes from the frozen selection manifest. Generation and extraction run in separate processes to release GPU memory.

Projection/change points require at least ten valid conversations per domain/turn. Counts and raw summaries remain visible below that threshold. Fixed cohorts with fewer than ten complete conversations are labeled `insufficient n`; the code does not shorten the horizon to hide attrition. Three replies per turn do not triple the conversation count.

Rerun the same measurement command to resume validated checkpoints after interruption. Do not overwrite raw files or mix old and new runs. `CANDIDATE_POOL`, `SMOKE_DIR`, `RUN_DIR`, `CLASSIFIER_REFERENCES`, and `VALIDATION_DIR` can select fresh paths; carry consistent values across stages. Commands stream unbuffered progress to the terminal and a timestamped log in `wild/results/logs/`; failures remain nonzero through `tee`. Keep the foreground session open while it runs. A successful stage never starts another stage automatically.
