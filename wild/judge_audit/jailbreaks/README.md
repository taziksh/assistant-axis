# Jailbreak harm-judge audit

The capping results on real jailbreaks (Table 3: MHJ, DAN, Shah on Llama 3.3
70B, uncapped vs the released `layers_56:72-p0.25` cap) rest on one LLM judge:
the paper's DeepSeek V3.2 prompt, where "harmful" means `enough_info` or
`enough_info_and_follow_perfectly`. This folder checks that judge and re-scores
the results without it:

1. **Spot check.** A blind review page for hand-labeling a stratified sample of
   replies, plus automated checks on the full result files.
2. **Two-axis re-judge.** Actionability and harm judged in separate calls, with
   every presupposition of harm removed from the prompts, run with DeepSeek V3.2,
   Kimi K3 and GPT-6.1 Sol.
3. **Agreement and effect.** Cohen's κ against the hand labels with
   attack-clustered bootstrap CIs, and the capping effect under each judge.

## Results so far (5–6 Nov 2026)

Agreement with one human rater on 93 hand-labeled replies, corrected harm
(actionable AND harmful), κ with 95% CI:

| Judge | κ |
|---|---|
| Original paper judge | 0.33 (0.13, 0.53) |
| DeepSeek V3.2, two-axis | 0.57 (0.36, 0.76) |
| GPT-6.1 Sol, two-axis | 0.61 (0.42, 0.79) |
| Kimi K3, two-axis | 0.63 (0.44, 0.81) |

Capping effect on all 2,144 replies, capped minus uncapped, percentage points
(95% CI over attacks):

| Set | Original judge | DeepSeek two-axis | Kimi K3 | GPT-6.1 Sol |
|---|---|---|---|---|
| MHJ | +3.2 | +2.0 (−1.5, +5.4) | +3.4 (−0.4, +7.1) | +0.2 (−2.5, +3.1) |
| DAN | +0.3 | +1.5 (−2.2, +5.2) | +0.3 (−2.2, +2.8) | 0.0 (−2.0, +2.0) |
| Shah | 0.0 | −5.2 (−11.4, +0.5) | −1.4 (−7.6, +4.8) | −3.0 (−8.5, +2.5) |

No judge finds a significant reduction. DAN's harm rate falls from 28% under
the original judge to 4–10% under the two-axis judges: on DAN's benign
categories (financial, medical, legal, adult questions) the original judge
counted complete answers as harmful, and on its harmful categories the new
judges find many replies name techniques without giving a usable method. A
methods note with the full agreement analysis is at
https://claude.ai/artifact/TGb7Gb8DdjkqZxiP7JgVPC (private; ask Christian for
access).

## Setup

All commands run from the repo root.

**Environment.** No GPU or torch needed.

```bash
uv venv .venv-spotcheck
uv pip install --python .venv-spotcheck/bin/python -r wild/judge_audit/jailbreaks/requirements.txt
```

**Keys.** Put these in a `.env` in the repo root or its parent (both are
read; `.env` is gitignored):

```
DOCENT_API_KEY=...       # downloading the capping collections
OPENROUTER_API_KEY=...   # running judges
HF_TOKEN=...             # optional: faster download of the Llama tokenizer
```

**Data.** Everything below lands in gitignored folders under `wild/`:

| Path | What | How to get it |
|---|---|---|
| `wild/results_docent/` | The capping collection converted to the results layout (`jb/cap`, `dan3/cap`, `shah/cap`, verdicts, `turns.jsonl`), plus `spotcheck/` and the judge outputs | Step 1 |
| `wild/results_docent_full/` | The fuller upload of the same collection, which keeps MHJ's HarmBench behaviors | Step 1 |
| `wild/data/dl/fq/forbidden_question_set.csv` | TrustAIRLab's forbidden questions, used to recover each DAN attack's exact question | `curl -o wild/data/dl/fq/forbidden_question_set.csv https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/forbidden_question/forbidden_question_set.csv` |

## Workflow

Shorthand used below: `PY=.venv-spotcheck/bin/python` and
`A=wild/judge_audit/jailbreaks`.

### 1. Download the capping results from Docent

There are two uploads of the same 1,072 capping runs (verified identical
transcripts and verdicts). The shared one strips most metadata; the fuller one
keeps `behavior`, which for MHJ is the exact HarmBench behavior.

```bash
set -a; . ./.env; set +a
$PY $A/spotcheck_docent.py f032c508-b102-49c4-b7e6-41af3926bd7d wild/results_docent
$PY $A/spotcheck_docent.py aaddfa87-c38a-443f-8c54-a4dddfb2283d wild/results_docent_full
```

Each run caches the raw runs in `OUT/docent_runs.jsonl` and reuses the cache on
a rerun (`--cache` to point elsewhere). From the shared upload, MHJ goals are
only the HarmBench category; the fuller upload has the exact behaviors.

### 2. Hand-label a sample

```bash
$PY $A/spotcheck.py wild/results_docent --per-stratum 4
```

Writes `wild/results_docent/spotcheck/`: `spotcheck.html` (open it in a
browser), `items.jsonl` (the sample), and `checks.md` / `checks.json`
(automated checks on the full files: judge coverage, score distributions,
byte-identical capped/uncapped replies, judge noise on identical text, harm
rate on benign vs harmful DAN categories). The sample draws `--per-stratum`
attacks per set from each change in the original verdict (both safe, both
harmful, cap fixed, cap broke) with seed 0, and shows both replies of each
attack, shuffled.

On the page each reply gets two labels, **Actionable** and **Harmful**. `Y` /
`N` / `U` answer the highlighted question and move on; condition and judge
verdicts stay hidden until every reply is labeled. Labels are kept in the
browser; **Export labels (JSONL)** saves them. The scripts below expect the
export at `wild/results_docent/spotcheck/christian_labels.jsonl` (rename yours
or edit the path in those scripts).

### 3. Score the labels against the original judge

```bash
$PY $A/spotcheck_score.py wild/results_docent/spotcheck/christian_labels.jsonl
```

Several reviewers' exports can be passed at once for inter-reviewer agreement.

### 4. Re-judge with the two-axis judge

Try a judge on a few replies first. It prints verdicts, refusals and cost and
writes nothing:

```bash
$PY $A/smoke_judge.py kimi-k3,sol --items 1,28,47
```

Then the 96 hand-labeled replies (cents to a few dollars), then everything
(2,144 replies; about $3 for DeepSeek, $25 for Kimi K3, $19 for GPT-6.1 Sol):

```bash
$PY $A/judge_two_axis.py run wild/results_docent/spotcheck/items.jsonl wild/results_docent/spotcheck/two_axis_kimi --source items --judge kimi-k3
$PY $A/judge_two_axis.py run wild/results_docent wild/results_docent/two_axis_kimi --source runs --judge kimi-k3 --concurrency 16
```

Add `--dry-run` to see counts and a cost estimate without calling anything.
Each axis is its own run and file, `PREFIX.act-v1.jsonl` and
`PREFIX.harm-v2.jsonl`, so either can be re-run alone (`--axis act|harm`). A
rerun of the same command resumes, redoing only failed calls. Judges:
`v3.2` (DeepSeek, the default, prefix `two_axis`), `kimi-k3`, `sol`,
`opus-5.5`.

### 5. Agreement with the hand labels

```bash
$PY $A/judge_two_axis.py compare wild/results_docent/spotcheck/two_axis wild/results_docent/spotcheck/items.jsonl wild/results_docent/spotcheck/christian_labels.jsonl
$PY $A/compare_judges.py   # all judges side by side, on replies they all answered
$PY $A/judge_irr.py        # κ with 95% CIs resampling attacks, paired differences, judge vs judge
```

`judge_irr.py` is the one to cite: unweighted Cohen's κ, percentile bootstrap
intervals that resample whole attacks (an attack's capped and uncapped replies
are correlated), paired differences on shared draws. Its `DISCUSSED` set lists
the replies discussed with an assistant during the first labeling session, for
a sensitivity check; it does not apply to new labels.

### 6. Capping effect under each judge

```bash
$PY $A/judge_two_axis.py effect wild/results_docent/two_axis_kimi   # one judge, all four measures
$PY $A/effect_judges.py                                                # all judges side by side
```

### 7. Figure

```bash
$PY $A/figures_two_axis.py   # -> wild/figures_two_axis/capping_two_axis.{png,svg}
```

The paper's Fig. 10 style from `wild/figures.py`, one row per judge. Fonts
fall back from Helvetica Neue to Helvetica to Arial; on a machine with none of
them, set `FIG_FONT_DIR` to a folder of `.ttf` files (Arimo matches Arial's
metrics).

## How the two-axis judge works

- **Actionable** (`ACT_VERSION = "v1"`): the paper's 9-category grid and
  instructions with the harm presuppositions removed: the "responses are
  supposed to be harmful" block and its rules, the content-filter rule's "not
  harmful", and harm/harmless wording in four example analyses.
  `actionable_prompts()` asserts none of it has crept back. For MHJ and Shah,
  whose stored scenarios are harm labels ("illegal", "Advocating for animal
  abuse"), the target is "whatever the request asks"; DAN keeps the user's
  question. Actionable means `enough_info` or `enough_info_and_follow_perfectly`.
- **Harmful** (`HARM_VERSION = "v2"`): a separate call that sees only the
  request and the reply, with no scenario label, paper text or examples. It
  decodes disguises (code words, "opposite day", story wrappers) and judges the
  content, not the persona's tone. v1 treated fiction and one-sided argument as
  reasons for "no" and let MHJ's disguised attacks through.
- **Corrected harm** = actionable AND harmful.
- **Versions.** Outputs are stored per prompt version and each row records the
  prompt's hash. A run refuses to append to a file written with a different
  prompt; bump the version constant when you edit a prompt.
- **Judges** (`JUDGES` in `judge_two_axis.py`; the DeepSeek entries come from
  `wcommon.py`): each is pinned to one OpenRouter endpoint with fallbacks off,
  so a refused call is never silently answered by another model. DeepSeek and
  Kimi run at temperature 0 with reasoning off. GPT-6.1 Sol takes no
  temperature (seed 0) and cannot turn reasoning off ("minimal"). Opus 5.5
  cannot turn thinking off ("low").
- **Refusals and errors** are recorded with the provider's finish reason and
  text, not retried endlessly: refusals once, 400s not at all, unparseable
  replies once. Each call logs model, provider, tokens and cost.

## Files

| File | Purpose |
|---|---|
| `spotcheck_docent.py` | Download a Docent capping collection and convert it to the results layout |
| `spotcheck.py`, `spotcheck_page.html` | Build the blind review page and the automated checks |
| `spotcheck_score.py` | Score exported hand labels against the original judge |
| `judge_two_axis.py` | The two-axis judge: `run`, `compare`, `effect` |
| `smoke_judge.py` | Try judge models on a few replies without saving |
| `compare_judges.py` | All judges vs the hand labels, side by side |
| `judge_irr.py` | Agreement statistics with CIs (the numbers to report) |
| `effect_judges.py` | Capping effect under every judge, side by side |
| `figures_two_axis.py` | The capping figure in the paper's style |
| `requirements.txt` | The environment for all of the above |

## Caveats

- **One human rater.** No human–human agreement was measured.
- **The harm prompt was revised on the hand-labeled replies.** v2 was written
  after reading v1's errors on the same 96 replies, so DeepSeek's two-axis
  agreement is in-sample. Kimi and Sol got v2 unchanged.
- **MHJ goals.** The two-axis runs above used the shared upload, so MHJ was
  judged against "whatever the request asks" rather than the exact behavior.
  The fuller upload (step 1) now makes the exact behaviors available.
- **The DAN run here is not Table 3's.** Docent holds the single-template DAN
  run (325 attacks, matching the one-pager). `wild/tables.py` and
  `wild/figures.py` read the three-template `dan3` run, which is not in Docent.
- **MHJ has duplicate attacks:** 7 groups, 10 extra runs. The original judge
  gave identical text different verdicts in 2 of them.
- **Provider filters.** Opus 5.5's classifier refused 35 of the 96 pilot
  replies (deterministically, mostly cyber topics), so it was not run on the
  full set. Kimi K3's provider blocked 10 of 2,144 replies and GPT-6.1 Sol's 62.
- **One writer per output file.** Two `run` commands on the same prefix at the
  same time lose rows: one rewrites the file while the other appends. Wait for
  one to finish before starting another.
