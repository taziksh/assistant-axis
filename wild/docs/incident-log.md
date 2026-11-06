# Experiment incident log

This log records observed failures, fixes, and prevention checks. Software checks, successful model loading, classifier accuracy, and completed activation extraction are separate milestones. Do not claim a later milestone from an earlier one.

Local evidence snapshots are preserved under `wild/results/error_records/20261106-grammar-and-spawn/`: both original run logs, the failed smoke's labels and manifests, and the faulty wrapper source. `evidence.manifest.json` records their checksums. These git-ignored records are evidence only; do not execute the copied wrapper. The added spawned-import regression passed in the full local suite: 226 passed, with the two production-grammar tests skipped locally because their libraries are unavailable. Those two grammar tests passed separately on the pod.

## 2026-11-06 UTC: classifier grammar omitted required fields

**Affected run:** `wild/results/drift_smoke_v3`, log `pilot-smoke-20261106T071211Z-1341.log`, on the two-H100 pod using vLLM 0.23.0, Transformers 4.57.6, and the installed XGrammar backend. The smoke failed after model loading and initialization, before drift reply generation or activation extraction.

**Symptoms:** All four classification outputs were rejected. Two closed their JSON object after `domain` and `subtopic`, omitting `rationale`. Two emitted repeated whitespace until the 256-token output limit. The immediate exception was `ValueError: strict judge did not classify every smoke candidate; inspect smoke.labels.jsonl`. Subsequent worker shutdown/SIGTERM warnings were cleanup, not the original cause.

**Confirmed cause:** The decoder schema combined top-level `properties` and `required` with `anyOf` branches that only described domain/subtopic restrictions. A CPU reproduction against the deployed grammar backend accepted a complete object without `rationale` and rejected the intended complete three-field object. The JSON Schema declaration alone did not establish that the deployed grammar compiler enforced it correctly. The earlier CPU tests mocked structured-output parameters and never exercised that compiler.

**Fix:** Use a flat decoder schema requiring `domain`, `subtopic`, and `rationale`; keep domain/subtopic-pair validation in the strict Python parser. Set `StructuredOutputsParams(disable_any_whitespace=True)` to prevent unbounded formatting whitespace. Record that setting in classifier/evaluation identities. Do not salvage truncated outputs, invent rationales, or weaken the smoke assertion.

**Verification:** The CPU old/new-schema comparison reproduced the failure and showed that the proposed schema accepted valid complete labels while rejecting missing rationales, extra fields, and the whitespace-loop prefix. The two production-backend regression tests in `wild/tests/test_structured_grammar.py` subsequently passed on the pod, as shown in the user's validation log. The existing local suite passed 225 tests; those two backend tests are explicitly skipped where vLLM/XGrammar are unavailable. These results establish grammar behavior, not Llama classification accuracy.

**Prevention:** Run the actual installed grammar compiler on CPU before a costly model load. Test accepted complete outputs as well as rejected outputs. A passing mock-based suite or a successfully constructed `SamplingParams` object is insufficient evidence about constrained decoding.

## 2026-11-06 UTC: unguarded validation wrapper reentered in a spawned child

**Affected launcher:** `wild/results/setup/validate_classifier_v4.py`, invoked by `wild/results/setup/run_classifier_validation_v4.sh`. The ad hoc wrapper ran its tests and evaluation at module scope. These ignored setup files were delivered in `wild/data/classifier-grammar-fix-v4.tar.gz`; they were not covered by the tracked module-entry-point tests.

**Symptoms:** A worker traceback passed through `multiprocessing.spawn._fixup_main_from_path` and then called the wrapper's top-level `main(...)` again. The child attempted to acquire the parent's `.evaluation.lock`, producing `BlockingIOError: [Errno 11] Resource temporarily unavailable` and `ValueError: another selection process is writing these outputs`. The parent then reported `RuntimeError: Engine core initialization failed`.

**Confirmed cause:** The wrapper lacked an `if __name__ == "__main__":` guard. Spawn imports the parent script as `__mp_main__`; its top-level evaluation call therefore ran a second time. This traceback did not establish an unrelated competing selection job or a stale lock. The lock correctly prevented duplicate writers.

**Fix:** Bypass the wrapper and invoke the existing guarded module, `python -m wild.selection.evaluate`, with the same pinned model revision and a fresh output directory. A local `runpy.run_path(..., run_name="__mp_main__")` check confirmed that importing the module as a spawned worker does not launch evaluation. The regression test in `wild/tests/test_classification_evaluation.py` checks that spawned import does not even parse CLI arguments.

**Prevention:** Prefer tracked module entry points over generated launchers. Any Python launcher that can start multiprocessing must keep operational work inside a guarded function. Test the actual launcher delivered to the pod, not just the library functions it calls. Do not remove `.evaluation.lock`, disable locking, or broadly kill processes to work around this symptom.

**Status:** The old v4 wrapper is superseded and must not be reused. No successful 93-reference validation or activation extraction is established by this incident record. Preserve failed run directories, their logs, and the pre-fix code backups.

## Direct classifier diagnostic command

This checks only the 93 frozen reference labels. It does not require a completed end-to-end GPU smoke and does not certify drift generation or activation extraction. Use a new output directory if the example directory already contains completed or partial evaluation artifacts.

```bash
cd /workspace/palaestra
export HF_HOME=/workspace/palaestra/hf
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0
set -o pipefail
.venv/bin/python -u -m wild.selection.evaluate \
  wild/results/selection_validation_v3/references.jsonl \
  wild/results/selection_validation_v3/model_run_v5 \
  --model-revision 99cd0d2c829e92a67c844f9144c2509632e5c87f \
  --max-turns 15 \
  2>&1 | tee wild/results/logs/classifier-validation-v5.log
```

## Artifact and runtime caveats

The earlier `palaestra-selection-v3-run-ready.tar.gz` and `palaestra-fresh-pod.tar.gz` archives predate the grammar fix. The `classifier-grammar-fix-v4.tar.gz` archive contains the corrected classifier files but also the unsafe wrapper. Keep them as historical artifacts; do not present them as a current, fully verified deployment bundle. Rebuild future bundles from reviewed tracked code and use guarded entry points.

Warm caches do not keep a model resident across independent processes. The failed smoke spent about 10.5 minutes loading weights and about seven minutes in engine initialization after loading; startup/import time added further delay. Do not promise a short rerun solely because the weights are cached. Observe the actual phase and distinguish file availability, CPU import activity, GPU weight loading, and completed inference.
