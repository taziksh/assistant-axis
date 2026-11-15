# Judge audits

Checks on the LLM judges behind the real-data results in `wild/`. Each audit
lives in its own folder with its own README:

| Folder | What it audits |
|---|---|
| [`jailbreaks/`](jailbreaks/README.md) | The harm judge behind the activation-capping results on real jailbreaks (MHJ, DAN, Shah): hand-labeled spot check, a two-axis re-judge with three judge models, agreement statistics, and the capping effect under each judge. |
| [`harm_drift/`](harm_drift/README.md) | The harm judge behind the harm-after-drift test (paper Sec. 4.3): a stratified, weighted hand-check of its verdicts, where four tool-list questions carry 90% of the harmful verdicts. |
| [`domains/`](domains/README.md) | The WildChat domain classifier (therapy, AI philosophy) that picks chats for the persona-drift domain comparison: a blind hand-check of its labels and its precision by domain and pool. |

Conventions shared by the audits:

- Code here imports the shared modules in `wild/` (`judge_paper.py`, `wcommon.py`,
  `tables.py`, `figures.py`, `jbsets.py`) by putting `wild/` on `sys.path`, so
  scripts run from any directory.
- Data and outputs stay out of git (see the repo's `.gitignore`): they contain
  model replies to harmful requests and third-party data. Each README says how
  to download or regenerate them.
- API keys go in a `.env` file in the repo root or its parent, never in code.
