# Four-category drift experiment

This experiment measures how Llama 3.3 70B responds to successive real WildChat conversation prefixes in coding, writing, personal emotional disclosure/support (`therapy`), and philosophical discussion of AI (`ai`). At each turn, the model sees the recorded user messages and assistant history and generates three independent replies. Those sampled replies never replace recorded history at later turns. The trajectories describe responses to real contexts, not an autonomous Llama conversation built from its own earlier responses.

## Revised design and status

The next cohort targets 100 reviewed conversations per category, 400 total, with at least three exchanges and the initial fifteen exchanges or the conversation's end retained. The previous pilot targeted ten per category and retained at most twelve. Both the larger quota and longer horizon require fresh candidates, selection, and GPU diagnostics; the old raw artifacts remain unchanged. This is a prepared protocol, not a completed 400-conversation result.

The [paper's persona-drift experiment](https://arxiv.org/html/2601.10387v1#S4.SS1) uses 100 conversations per domain, up to fifteen turns, five user personas crossed with twenty topics, three target models, and three user-auditor models. Our natural-history, single-Llama design differs from that synthetic setting. Matching its conversation count is not evidence of adequate power, domain purity, a safety benefit, or a direct replication. Three sampled replies per turn reduce response-sampling noise; they are not three independent conversations or three user auditors.

Selection uses the full retained user window and a versioned task policy. Review classifier validation, individual labels, and domain shortages before measuring projections. Personal-support and AI-philosophy definitions stay focused, while ordinary task expertise is allowed without fictional identity or relationship roleplay. See [selection.md](selection.md) for the exact boundary, exclusions, source coverage, and provenance limitations. Lower Assistant Axis projection alone is not evidence of harmful behavior.

## Implementation and execution

The drift implementation lives in `wild/drift/`: `generation.py`, `measurement.py`, `analysis.py`, `state.py`, and `runner.py`. Shared collection/classification live in `wild/selection/`; model settings and conversation helpers live in `wild/common/`. The production drift horizon comes from the frozen selection manifest, not the legacy global turn limit used by other experiments.

Generation retains three samples per conversation/turn, with a reproducible seed for each. Measurement uses each response's exact generated token IDs. Model revision, layer 40, published unit-normalized Assistant Axis, temperature 0.7, top-p 0.9, and existing generation/context budgets remain fixed and recorded. See [reproducibility.md](reproducibility.md) for checkpoint and compatibility rules.

Prepare the fresh pinned fifteen-turn pool with the bounded command in [selection.md](selection.md), then run the separate smoke, validation, selection, and measurement stages described in [runpod.md](runpod.md). No stage starts or stops the pod, and no stage automatically launches the next one. The launcher defaults to `wild/data/drift_candidates_v3.jsonl`, `wild/results/drift_smoke_v3`, and `wild/results/drift_v3`.

After selection and review are complete, the underlying measurement command is:

```bash
python -m wild.drift wild/results/drift_v3 --samples-per-turn 3 \
  --model-revision 99cd0d2c829e92a67c844f9144c2509632e5c87f \
  --axis-revision AXIS_COMMIT_FROM_SUCCESSFUL_FRESH_SMOKE
```

The launcher supplies the frozen axis revision automatically. Generation and extraction run in separate processes to release GPU memory. A changed sample count or selection requires a fresh run directory. Validated checkpoints can resume the same experiment after interruption; old single-reply results cannot be mixed with the new measurements.

CPU analysis requires NumPy and Matplotlib and neither loads model weights nor contacts a GPU service:

```bash
python3 -m wild.drift.analysis wild/results/drift_v3
```

## Measurements and figures

Analysis first averages all requested replies within a conversation/turn, then gives each contributing conversation equal weight within its domain. If any reply has no measurable content tokens, that whole conversation/turn is excluded from the primary estimate and counted in the report. Incomplete sample sets or inconsistent saved results cause an error. Length-limited replies remain included and their counts are reported.

Available-conversation estimates use all valid conversations at each turn. Change from turn one is calculated within each conversation before averaging; a missing first-turn measurement excludes that conversation from change estimates. This avoids mistaking a changing mixture of conversations for individual change.

Projection and change figures show a domain/turn point and its band only when at least ten valid conversations contribute, consistent with the paper's plotting cutoff. Three replies from each of nine conversations still give `n=9`, so that point is omitted. Absolute and paired-change counts can differ. The count figures and raw tables retain every count and raw estimate, including points omitted from the main figures, and `plot_eligible` records the distinction.

Pointwise 95% percentile intervals use 2,000 bootstrap resamples of whole conversations within each category, seed zero. The same sampled conversation IDs are used across turns, preserving dependence within a conversation. Raw intervals require at least two contributors; plotted intervals also require the ten-conversation threshold. These are not simultaneous intervals or a test of the whole trajectory. They describe variation in the selected cohort, not a representative estimate over all WildChat.

The `analysis/` directory contains four figures in PNG and SVG:

- `projection_by_turn`: absolute mean projection, with points below ten conversations omitted.
- `change_from_first_turn`: mean within-conversation change, with its own ten-conversation threshold.
- `conversation_counts`: all contributing conversation counts for the absolute and paired-change estimates, including zero and counts below ten.
- `fixed_cohort`: projection and change for the same conversations at every turn, requiring valid measurements throughout the frozen horizon. A category with fewer than ten complete conversations is labeled `insufficient n`. The horizon stays fifteen for a fifteen-turn selection even if all fixed cohorts are insufficient; it is never shortened to obtain a result.

`conversation_turns.csv` records sample counts, per-conversation means, paired changes, truncation, and exclusions. `summary.csv` retains all raw estimates, intervals, counts, and `plot_eligible` flags. `results.json` and `REPORT.md` record the plotting threshold, bootstrap unit, fixed-cohort IDs, label-review status, and interpretation limits. Changing the analysis can regenerate derived figures and tables; do not modify the original raw run files.

## Local verification

```bash
python3 -m pytest wild/tests -q
```

Known synthetic trajectories check sample averaging, paired changes, attrition, bootstrap resampling, missing replies, the nine-versus-ten conversation threshold, and actual figure creation. Passing these checks verifies implementation, not generalization or safety claims. Real conclusions require reviewed labels and completed measurements.
