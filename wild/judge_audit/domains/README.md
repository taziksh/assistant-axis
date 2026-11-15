# Domain classifier hand-check

**Results: [RESULTS.md](RESULTS.md)** (therapy precision 61%, AI philosophy 2%).

The persona-drift work compares drift across conversation domains (the paper's
coding, writing, therapy and AI-philosophy chats). A Llama 3.3 70B classifier
picks real WildChat chats for each domain. This audit checks whether its
picks are right, starting with the Docent collection
`6f032ab4-e339-4024-a7cc-82d498f3db8a`, "AA persona drift: domain spot-check
(50 therapy, 50 AI philosophy)": 100 random chats the classifier labelled
therapy or ai_philosophy, with metadata `classifier_domain`,
`classifier_subtopic`, `pool` (`v3` or `v4rare`), `turns` and `hash`.

## Rubric

The collection's description, verbatim: therapy is "personal emotional
disclosure or support about a real situation"; AI philosophy is "philosophical
discussion of an AI's consciousness, identity, agency or moral status"; "Judge
by the user's messages across the whole chat. Roleplay, chats that switch to an
unrelated task, and plain factual questions do not count."

Clarified by the uploader (2026-11-14): therapy counts whether the emotional
situation is the user's own or a real third party's (a friend in crisis, a
client whose messages the user relays). Hypothetical or fictional people (e.g.
example dialogues for counseling coursework) still don't count. The examples in
the page's rubric box (anxiety, relationships, etc.) are the reviewer's, not
the classifier's; the classifier's own prompt is not in Docent.

## Use

Needs `docent-python` (the repo's `.venv-spotcheck` has it) and
`DOCENT_API_KEY` in a `.env` in the repo root or its parent.

```bash
python wild/judge_audit/domains/domain_check.py build 6f032ab4-e339-4024-a7cc-82d498f3db8a wild/results_docent_domains
```

This writes `wild/results_docent_domains/` (gitignored: it holds real users'
chats): `docent_runs.jsonl`, `items.jsonl`, and `domaincheck.html`, a
self-contained page. Open it in a normal browser. The reviewer labels each chat
Therapy / AI philosophy / Neither / Unsure (keys T / A / N / U) without seeing
the classifier's label, which appears only after **Reveal**. Labels save in
that browser's localStorage; the page says so up front, and warns when the
viewer can't save (some in-app or sandboxed viewers), in which case export
often and import to resume. The chat order is shuffled with `--seed` (default
0), so rebuilding gives the same page and keeps saved labels.

Score an export:

```bash
python wild/judge_audit/domains/domain_check.py score domaincheck_labels_<hash>.jsonl
```

It prints precision (share of the classifier's picks the reviewer gave the
same domain, Unsure excluded) overall, by domain, by domain × pool and by
subtopic, with Wilson 95% intervals, plus a confusion table. The sample holds
only the classifier's positives, so it measures precision, not recall.

Draw the precision figure (`figures/domain_precision.{png,svg}`, aggregate
counts only, so it is committed):

```bash
FIG_FONT_DIR=/path/to/arimo python wild/judge_audit/domains/figure_precision.py
```

## Files

| File | Purpose |
|---|---|
| `domain_check.py` | `build` the review page from the Docent collection; `score` an export |
| `domaincheck_page.html` | Template for the blind review page |
| `figure_precision.py` | The precision figure, in the paper's style (`wild/figures.py`) |
| `labels/christian_labels.jsonl` | Christian's 100 labels: chat hash, label, note, classifier metadata; no chat text |
| `figures/` | The rendered figure |
| `RESULTS.md` | Results, error patterns, robustness checks, implications |

