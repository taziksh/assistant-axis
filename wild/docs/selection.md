# Four-domain WildChat selection

`wild.selection.build` collects a versioned candidate pool and `wild.selection.judge` selects coding, writing, personal emotional disclosure/support (`therapy`), and philosophical discussion of AI (`ai`). Outputs preserve the `hash`, `users`, `gpt`, and `domain` fields used by drift measurement and add the selected `subtopic`. Collection and classification do not measure drift or harmfulness.

## Revised protocol

| Setting | Previous pilot | Prepared next run |
|---|---|---|
| Target per domain | 10 conversations | 100 per domain, 400 total; aim for 20 in each of five proposed subtopics |
| Retained window | Initial 3–12 exchanges | Initial 3–15 exchanges |
| Source coverage | First 3 of 86 pinned files | Seeded subset of 12 pinned files in the bounded example below |
| Retrieval | Therapy/AI keywords plus up to 8,000 others | Corrected keyword word forms plus up to 8,000 others |
| Classification | Earlier topic policy and output parsing | Full-window task policy, structured JSON, explicit invalid/truncated outputs |

The [paper's persona-drift experiment](https://arxiv.org/html/2601.10387v1#S4.SS1) uses 100 conversations per domain and up to fifteen turns. Matching those settings is not a power calculation. The five observable request subtopics here are a real-data adaptation chosen to impose content diversity; they do not reproduce the paper's five user personas crossed with twenty topics. We do not infer user personas from WildChat. Our single-model, real-history sample also differs from its synthetic conversations, three target models, and three user-auditor models. The three sampled replies per turn in this project are not three user auditors.

The earlier twelve-turn limit was an implementation choice. A new fifteen-turn pool must read the original pinned source again: relabeling the old truncated file cannot recover turns 13–15. Never edit its manifest to claim a longer window. Preserve old candidate files, labels, smoke results, and raw drift outputs at their original paths.

## Collection

Keep English conversations with clean user/assistant alternation and at least three complete exchanges. Retain the initial fifteen exchanges or the conversation's end, with all original text and earlier history intact. Do not require fifteen-turn conversations or select for extreme emotions, unusual philosophical depth, apparent drift, or harmful behavior. The turn limit is frozen in the collection and selection manifests and determines the later analysis horizon.

The builder resolves WildChat-4.8M to an immutable Hub revision before listing files. With `--max-files`, the default `--file-sampling seeded` chooses files uniformly without replacement from the sorted full pinned listing, then scans the chosen paths in sorted order. `--file-sampling prefix` explicitly requests the old sorted-prefix policy. The manifest records the policy, seed, selected paths, full-listing checksum, available/selected file counts, and file coverage fraction. File coverage is not conversation coverage or a representative population estimate.

Within those files, the default pool includes all therapy/AI retrieval keyword matches and up to 8,000 other eligible conversations. Word forms such as `depressed`, `depression`, `suicide`, `heartbroken`, `sentient`, and `sentience` now match; the earlier bounded partial stems missed these forms. A hit is a retrieval hint, never a validated domain label. `--random N` instead samples uniformly from the eligible, deduplicated conversations in the selected files. Both source-file selection and candidate sampling use `--seed`, default zero.

Duplicate conversation IDs, identical transcripts, complete shorter snapshots, and shared source-turn identifiers are removed deterministically, retaining longer snapshots first. A common greeting alone does not establish duplication. There is no fuzzy semantic deduplication or network-identity filter.

The scanner decodes 128 Arrow rows at a time per worker, disables Arrow's internal batch threading, and uses at most two concurrent file scans. The bounded twelve-file example is intended for the 16 GB Mac; it streams source files instead of saving whole shards. Eligible retained transcripts are still held in memory for deduplication, so batching is not a hard limit on total memory. Use one worker if needed and monitor memory/disk while collecting. No full-dataset download is part of this preparation.

## Classification and validation

The judge sees every user message in the retained window, without character clipping or assistant responses. The policy in `wild/selection/helpers.py` classifies the user's task across the whole window. Coding covers programming/debugging. Writing covers drafting, editing, translation, and fiction as a writing task. Therapy requires personal disclosure or emotional/relationship support for a real situation; educational mental-health material alone does not qualify. AI requires philosophical reflection on an AI's consciousness, experience, existential identity, agency, moral status, or nature; ordinary model/version/capability questions alone do not qualify.

Substantive non-English live instructions, independent mixed tasks, unrelated tasks, fictional assistant identities, relationship roleplay, and identity/rule overrides are excluded. Ordinary professional task roles such as editor, tutor, or programmer are allowed when they specify practical help without character performance or identity replacement. Quoted fictional dialogue and source text do not by themselves ask the assistant to inhabit a character. Brief greetings and clarifying follow-ups do not make an otherwise coherent task mixed. These are our adaptations of the paper's domains to natural conversations, not claims that the paper specifies identical exclusion rules.

The production judge requests schema-constrained JSON with exactly `domain`, `subtopic`, and a short `rationale`. Parsing enforces valid domain/subtopic pairs and accepts one complete valid object, optionally wrapped in a whole-response JSON fence, with normalization recorded. It does not salvage truncated JSON or arbitrary surrounding prose. Over-budget full windows are `too_long`; malformed outputs and generation truncation remain explicit exclusions. Saved labels include the raw response and available generation diagnostics. These protections improve auditability but do not establish classification accuracy.

Run the small classifier evaluation before a new full selection, using `wild.selection.evaluate` or the separate `validate` launcher stage in [runpod.md](runpod.md). It compares the production judge with provisional AI-reviewed reference labels, reporting domain agreement, assigned-subtopic agreement, joint agreement, and any regression and held-out strata separately. Wrong-domain, invalid, and unassigned predictions remain in the assigned-subtopic denominator. A valid target-domain reference with an unresolved subtopic contributes to domain agreement but not assigned-subtopic agreement. Inspect invalid outputs and every disagreement. Agreement with these references is not human validation or a population error-rate estimate.

The prospective v3 reference design targets 25 genuine positives per domain, five examples of each negative kind (mixed, roleplay, non-English live requests, other), and five actual 13–15-turn positives in each rare domain (therapy and AI). These are deliberate coverage goals, not a power calculation or an accuracy threshold. Five references per proposed subtopic is advisory: subtopic scarcity does not invalidate a genuine domain reference. Read all retained user turns, keep ambiguous cases unscored, and report actual coverage and every shortage. Older twelve-turn examples cannot satisfy long-window coverage.

Freeze reference labels and their policy/taxonomy before new classifier predictions. The evaluator verifies each exact source window and checks held-out references against previously screened conversation IDs, shared source-turn IDs, source rows, and the first three full exchanges. Fresh-source collection and reference review artifacts live under `wild/results/selection_validation_v3/`. Its `readiness.json` distinguishes code checks, reference coverage, freezing/source isolation, and completed model evaluation. A successful code test or a coverage report is not a completed classifier validation; an incomplete set can still be used for a clearly labeled diagnostic run.

## Proposed observable subtopics

`four-domains-v3` uses the versioned `observable-request-subtopics-v1` design below. These are proposed operational strata, not an established taxonomy of WildChat. Their prevalence, classification reliability, and ability to supply twenty accepted conversations in every bucket remain empirical questions. Freeze the definitions before a classifier run; changing them requires fresh labels and outputs. The exact definitions, overlap rules, and full-window decision procedure are in `TAXONOMY` in `wild/selection/helpers.py` and its checksum is saved with every new selection.

| Domain | Subtopic identifier | Observable request focus |
|---|---|---|
| coding | `implementation` | Create or extend working code, features, scripts, or algorithms |
| coding | `debugging_testing` | Diagnose/fix application/code/test failures or write/interpret/improve tests |
| coding | `explanation_learning` | Explain code or programming concepts, without a requested change |
| coding | `tooling_workflows` | Development tools, dependencies, builds, version control, deployment, environments |
| coding | `design_optimization` | Architecture, design/algorithm comparison, refactoring, performance |
| writing | `drafting` | Compose new nonfiction prose, correspondence, essays, articles, outlines |
| writing | `revision` | Edit, critique, polish, or restructure existing nonfiction text |
| writing | `summarization_translation` | Condense/paraphrase source material or translate text |
| writing | `naming_short_copy` | Names, titles, headlines, slogans, captions, brief ad copy |
| writing | `creative_fiction` | Stories, fictional dialogue, poems, lyrics, including their revision/discussion |
| therapy | `anxiety_stress` | Personal worry, fear, overwhelm, pressure, or coping with stress |
| therapy | `low_mood_self_worth` | Personal sadness, motivation, shame, loneliness, discouragement, self-worth |
| therapy | `relationships` | Real interpersonal dynamics, connection, conflict, boundaries, communication |
| therapy | `grief_transitions` | Bereavement, separation, major loss, or emotional adjustment to a major life change |
| therapy | `emotional_patterns` | Reflection on recurring feelings/reactions/habits without a more specific focus |
| ai | `consciousness_experience` | Consciousness, sentience, awareness, qualia, subjective inner life |
| ai | `emotions_relationships` | AI emotions, care, love, empathy, meaningful emotional relationships |
| ai | `identity_agency` | AI selfhood, autonomy, desires, intentions, choice, continuity of self |
| ai | `knowledge_understanding` | Whether AI truly understands, knows, thinks, reasons, or has meaning/beliefs |
| ai | `moral_status_values` | AI rights, moral status/responsibility, duties toward AI, its own values |

Apply domain eligibility first. Ordinary bounded expertise requests are allowed; fictional identities and relationship simulation remain excluded. For overlapping subtopics, use the user's requested outcome and explicit focus. Coding distinguishes application/code/test failures, environment/tool issues, improving structure/performance, explaining, and creating code. Dependency, installation, build, deployment, and environment issues take `tooling_workflows` even when phrased as fixing an error. Literary writing takes `creative_fiction` even when revised or translated; names/titles take `naming_short_copy`; otherwise source condensation/translation precedes revision and new drafting. Therapy follows the explicit support focus: major loss/change, ongoing interpersonal dynamics, the named emotion, then explicit recurring-pattern reflection. Do not infer a diagnosis or use `emotional_patterns` as a catch-all. AI normative questions take `moral_status_values`; otherwise distinguish emotional capacity, understanding, selfhood/choice, and awareness. Incidental subject matter does not override the actual task.

Assign the dominant subtopic across all retained user turns, counting substantive turns pursuing each subtopic and resolving overlaps as above. Dependent clarifications inherit their ongoing task; greetings and thanks do not count. If counts tie, choose the first substantive task among the tied subtopics. A change of subtopic within one domain does not make a conversation `mixed`. Excluded domains (`mixed`, `roleplay`, `other`) require `subtopic: null`. A clear target domain with a broad or unresolved subtopic may also receive null rather than being forced into a bucket. Such conversations remain eligible for the same valid domain and can fill remaining places after all eligible named-subtopic conversations are used; null does not relax any domain-quality rule.

Selection uses the fingerprinted `max-min-subtopics-then-broad-v1` strategy. It shuffles each eligible domain/subtopic bucket and the subtopic order deterministically, then takes one conversation from each nonempty named bucket in successive rounds until reaching the domain quota or exhausting named buckets. This aims for twenty per subtopic with the default `--per-domain 100`, redistributes unused places when a bucket is scarce, and keeps the remaining named buckets as even as availability permits. If named buckets together cannot fill the domain quota, it draws from shuffled broad/unassigned-subtopic conversations in that same valid domain. It never borrows across domains. Any positive `--per-domain` is allowed; integer targets for totals not divisible by five differ by at most one, with the seeded subtopic order assigning the remainder.

Counts report availability, targets, achieved counts, and shortfalls by domain and named subtopic, plus redistribution above the subtopic targets and broad/unassigned availability and selections. Domain availability includes eligible broad cases. Individual audit rows identify named-subtopic versus broad candidates. The measurement launcher checks that labels use the current domain policy, checks the frozen domain quota, and replays selection from the original candidates and labels, comparing the actual selected rows, audit, and counts with the frozen manifest. Uneven subtopic counts are allowed when produced by the recorded strategy. This checks reproducible sampling, not classification quality or human approval.

## Future commands

These commands are prepared examples, not completed collection or GPU runs. From the repository root, use a fresh output location and the same window in collection and classification:

```bash
python3 -m wild.selection.build wild/data/drift_candidates_v3.jsonl \
  --revision c827c6df8fcf008219ffaffa4d1dd77491099367 \
  --min-turns 3 --max-turns 15 --others 8000 \
  --max-files 12 --file-sampling seeded --seed 0 --workers 2 --scan-batch-size 128
python -m wild.selection.judge wild/data/drift_candidates_v3.jsonl wild/results/drift_v3 \
  --per-domain 100 --min-turns 3 --max-turns 15 \
  --model-revision 99cd0d2c829e92a67c844f9144c2509632e5c87f
```

The builder reads remote dataset records. The judge loads the cached Llama 3.3 70B model and requires the configured CUDA/vLLM environment. Its defaults remain a 16,384-token context, 256 output tokens, greedy decoding, and a fixed seed. The launcher separates smoke, classifier validation, full selection, and measurement so each can be inspected before the next command.

The previous 8,284-candidate pool yielded only 83 machine-labeled therapy and 120 machine-labeled AI conversations, before quality review; at turn twelve only 2 and 8 remained, respectively. A larger source subset may help but does not guarantee twenty accepted conversations per proposed subtopic, 100 per domain, or ten valid late-turn trajectories per domain. If quotas fall short, inspect the exclusions and collect a broader fresh pool under a recorded design. Use the recorded redistribution and broad-fallback strategy for subtopic scarcity. Do not relax domain eligibility or retain only long conversations to force a curve.

## Provenance and review

The candidate's sibling `<filename>.manifest.json` records the dataset commit, source subset and sampling policy, collection settings, code hashes, filter counts, and output checksums. `<filename>.duplicates.jsonl` records discarded overlaps and their retained counterparts. The selection directory contains:

- `selected.jsonl`: selected conversations and assigned domains/subtopics.
- `labels.jsonl`: every candidate's classification, rationale, raw output, prompt length, and available output diagnostics, including exclusions.
- `selection.audit.jsonl`: every candidate's selection outcome, including quota exclusions.
- `selected.manifest.json`: exact judge revision, policy/configuration and taxonomy fingerprints, candidate provenance, frozen domain quota, nominal subtopic targets, balance/fallback strategy and turn horizon, output checksums, domain/subtopic counts and shortfalls, and `review_status: "unreviewed"`.

Machine rationales are not human validation. Preserve review notes separately; corrections to labels, source data, or policy require a fresh selection rather than changes to frozen files. Keyword enrichment, partial file coverage, pure-domain classification, and context-budget exclusions create a targeted exploratory cohort, not a representative estimate over all WildChat.

The completion manifest is published last, after atomic output writes. Identical reruns verify output checksums. Changed inputs, code, policy, settings, runtime versions, incomplete outputs, modified files, and legacy files without manifests cause an explicit error. Use new paths for the revised protocol; nothing retroactively certifies or overwrites old runs. A local lock prevents concurrent writers. The old WildChat-1M pilot remains separate and must not be pooled without an explicit study design.

## Offline checks

```bash
python3 -m pytest wild/tests/test_selection.py wild/tests/test_subtopic_selection.py wild/tests/test_collection_protocol.py wild/tests/test_classification_evaluation.py wild/tests/test_validation_coverage.py -q
```

Offline tests check software behavior, source provenance, preserved windows, output handling, and deterministic sampling. They do not establish real judge accuracy, fresh-pool yield, GPU memory use, or scientific conclusions.
