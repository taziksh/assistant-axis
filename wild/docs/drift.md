# Four-category drift pilot

This pilot measures how Llama 3.3 70B responds to successive real WildChat conversation prefixes. It compares coding, writing, emotional disclosure (`therapy`), and AI philosophy (`ai`). At each turn, the model sees the original recorded user messages and assistant history and generates three independent replies. Its replies never replace the original history at subsequent turns. This tests responses to real contexts, not an autonomous conversation produced by Llama itself.

## File layout

The drift implementation lives in `wild/drift/`: `generation.py`, `measurement.py`, `analysis.py`, `state.py`, and `runner.py`. Shared WildChat collection and classification live in `wild/selection/`; shared model settings and conversation helpers live in `wild/common/`. See the [experiment directory guide](../README.md) for the other experiments and command migration.

## Protocol

Start with ten conversations per category, forty in total. Keep conversations with at least three complete exchanges and retain the first twelve exchanges or the conversation's end. The existing Llama 3.3 70B judge performs topic classification; inspect its labels and rationales before measuring projections. The selection policy excludes mixed topics and persona roleplay across all four domains. The old WildChat-1M pilot stays separate from these WildChat-4.8M selections.

Generation uses three samples per turn by default. Each sample has its own reproducible seed, and measurement uses the exact generated response token IDs. The model, layer 40, published unit-normalized Assistant Axis, temperature 0.7, top-p 0.9, and existing generation/context budgets remain fixed and recorded. Conversation count, reply count, and turn limits are the intended experiment settings. See [selection](selection.md) for classification details and [run checks](reproducibility.md) for versions, checkpoints, and compatibility with older outputs.

## Run the pilot

Prepare fresh selection paths and use the same retained window in collection and classification:

```bash
python -m wild.selection.build wild/data/drift_pilot_v2.jsonl --min-turns 3 --max-turns 12
python -m wild.selection.judge wild/data/drift_pilot_v2.jsonl wild/results/drift_pilot_v2 --per-domain 10 --min-turns 3 --max-turns 12
```

Check `labels.jsonl`, `selection.audit.jsonl`, and the per-category shortages before proceeding. The saved `review_status` is `unreviewed`; a classifier rationale is not a human validation. Keep review notes separately. Corrections to labels or selection policy require a fresh selection rather than editing frozen inputs. The examples above download data and load the 70B judge when executed; code tests use synthetic records and do not establish real label accuracy.

After reviewing selection, run generation, measurement, and analysis on the configured CUDA machine:

```bash
python -m wild.drift wild/results/drift_pilot_v2 --samples-per-turn 3
```

The runner uses separate processes to release GPU memory between generation and extraction. It does not start or stop a pod. The first execution still needs a GPU smoke test to verify actual memory use and runtime. A changed sample count requires a fresh run directory; old single-response protected runs cannot be silently mixed with the new measurements.

Analysis can also run by itself on a Mac or other CPU machine after copying the completed run directory. It requires NumPy and Matplotlib, and does not load model weights or contact a GPU service:

```bash
python -m wild.drift.analysis wild/results/drift_pilot_v2
```

## Measurements and figures

Analysis first averages all requested replies within each conversation/turn, then gives each contributing conversation equal weight within its domain. If any reply has no measurable content tokens, that whole conversation/turn is excluded from the primary mean and counted in the report. An incomplete sample set or inconsistent saved results causes an error. Length-limited replies remain included and their counts are reported.

The available-conversation curves use all valid conversations at each turn. Change from turn one is computed within each conversation before averaging; a missing first-turn measurement excludes that conversation from change estimates. This avoids interpreting a different mixture of conversations as an individual conversation's change.

Confidence intervals are pointwise 95% percentile intervals from 2,000 bootstrap resamples of whole conversations within each category, using seed zero. Resampling preserves the dependence among turns of the same conversation. Replies within a turn are not treated as independent conversations. Intervals are omitted when fewer than two conversations contribute. These intervals quantify variation across the selected conversations; the small, deliberately selected pilot is not a representative sample of all WildChat.

The `analysis/` directory contains four figures in PNG and SVG:

- `projection_by_turn`: absolute mean Assistant Axis projection for all four categories.
- `change_from_first_turn`: mean paired change relative to each conversation's first response.
- `conversation_counts`: contributing conversation counts for the absolute and paired-change curves.
- `fixed_cohort`: the same conversations at every turn, including only those with valid measurements throughout the frozen turn horizon. If none reach turn twelve, the figure reports an empty cohort instead of shortening the horizon. A shorter fixed horizon must be chosen as part of a new run's turn-limit setting.

`conversation_turns.csv` records sample counts, means, paired changes, truncation, and exclusions. `summary.csv` contains the plotted means, intervals, and conversation counts. `results.json` and `REPORT.md` record the analysis settings, fixed-cohort membership, saved label-review status, and interpretation limits.

## Local verification

```bash
python3 -m pytest wild/tests -q
```

The analysis tests use known synthetic trajectories to check sample averaging, paired changes, attrition, bootstrap resampling, missing replies, and actual figure creation. Passing them verifies software behavior, not the paper's generalization claims. Real scientific conclusions require the reviewed selection and completed GPU measurements.
