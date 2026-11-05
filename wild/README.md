# Real-chat experiments

The code is grouped by experiment and purpose. Run the Python commands from the repository root with `python -m ...`.

| Folder | Contents |
|---|---|
| `selection/` | WildChat collection, topic classification, and selection checks |
| `drift/` | Three-reply generation, exact-token measurement, and four-category plots |
| `harm/` | Harmful-question experiments, role openings, baselines, and harm judges |
| `capping/` | Capped generation, jailbreak datasets, capability checks, and case studies |
| `reports/` | Cross-experiment tables, paper-style figures, historical analysis, and Docent export |
| `common/` | Model settings, conversation helpers, and repository paths shared by experiments |
| `legacy/` | Single-reply generation, text-based measurement, and paper-transcript calibration |
| `scripts/` | Historical RunPod shell workflows |
| `docs/` | Selection, drift, and saved-run guides |
| `tests/` | Offline regression and integration tests |

Downloaded data, run outputs, and exported figures keep their existing locations at `wild/data/`, `wild/results/`, and `wild/figures/`. They remain outside Git.

## Current drift workflow

See the [pilot guide](docs/drift.md) for the full protocol. The main commands are:

```bash
python -m wild.selection.build wild/data/pilot.jsonl --min-turns 3 --max-turns 12
python -m wild.selection.judge wild/data/pilot.jsonl wild/results/pilot --per-domain 10 --min-turns 3 --max-turns 12
python -m wild.drift wild/results/pilot --samples-per-turn 3
```

Collection downloads the dataset. Classification and generation use Llama 3.3 70B on CUDA. Review the selection before running the measurement experiment. Completed measurements can be analyzed on a CPU machine:

```bash
python -m wild.drift.analysis wild/results/pilot
```

## Familiar command names

The old root-level Python files have moved. One dispatcher preserves their command names and arguments without keeping a launcher file for every script:

```bash
python -m wild --help
python -m wild build wild/data/pilot.jsonl --min-turns 3 --max-turns 12
python -m wild judge wild/data/pilot.jsonl wild/results/pilot --per-domain 10
python -m wild drift_run wild/results/pilot --samples-per-turn 3
```

For example, replace `python wild/harm_wild.py ...` with `python -m wild harm_wild ...` or `python -m wild.harm.generate ...`. Command aliases are listed in `wild/__main__.py`. Python imports use the new package paths; there are no duplicate implementations or module aliases.

The shell workflows moved to `wild/scripts/run.sh`, `harm_floor.sh`, `roles.sh`, and `shah.sh`. Their Python calls use the new modules. Their existing environment setup and pod-stop behavior remain; the dedicated drift runner does not manage pod lifecycle.

## Guides and tests

- [Conversation selection](docs/selection.md)
- [Four-category drift pilot](docs/drift.md)
- [Saved-run checks and resumption](docs/reproducibility.md)

```bash
python3 -m pytest wild/tests -q
```

Moving code changes the source fingerprints stored by protected runs. Completed schema-v2 outputs remain readable by the CPU analysis. An interrupted run with older code fingerprints must use its original checkout to resume; the new code will not silently adopt it.
