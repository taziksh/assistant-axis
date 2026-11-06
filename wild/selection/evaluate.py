"""Compare the production classifier with explicit real-conversation references."""

import argparse
from collections import Counter
import json
from pathlib import Path

from wild.common.settings import MODEL
from wild.selection import helpers as selection
from wild.selection import judge
from wild.selection import coverage

REFERENCE_STATUSES = {"ai_reviewed_provisional", "human_adjudicated"}
SPLITS = {"regression", "heldout", "coverage"}
INVALID_STATUSES = ("unparsed", "too_long")


def _artifact(manifest, name, base):
    entry = manifest.get(name, {})
    if not isinstance(entry.get("path"), str) or not entry["path"]:
        raise ValueError(f"reference manifest needs {name}")
    path = base / entry["path"]
    if not path.is_file() or selection.file_checksum(path) != entry.get("sha256"):
        raise ValueError(f"reference {name} fingerprint mismatch")
    return path


def _validate_label(domain, topic):
    return domain in selection.LABELS and (topic is None or domain in selection.DOMAINS and topic in selection.SUBTOPICS[domain])


def _source_isolation(cases, manifest, base):
    source = selection.read_jsonl(_artifact(manifest, "source_artifact", base))
    by_id = {row["hash"]: row for row in source}
    if len(by_id) != len(source):
        raise ValueError("reference source has duplicate conversation IDs")
    exclusions = json.loads(_artifact(manifest, "exclusion_artifact", base).read_bytes())
    required = {"conversation_hashes", "first3_exchange_sha256", "source_turn_id_json", "source_identities"}
    if not required <= exclusions.keys():
        raise ValueError("reference exclusions omit identity checks")
    old_ids = set(exclusions["conversation_hashes"])
    old_prefixes = set(exclusions["first3_exchange_sha256"])
    old_turns = set(exclusions["source_turn_id_json"])
    old_sources = {(row["source_file"], row["source_row"]) for row in exclusions["source_identities"]}
    for case in cases:
        row = case["conversation"]
        if by_id.get(row["hash"]) != row:
            raise ValueError("reference differs from its original source window")
        if case["split"] not in {"heldout", "coverage"}:
            continue
        prefix = selection.checksum(selection.json_bytes(list(zip(row["users"][:3], row["gpt"][:3]))))
        turns = {json.dumps(turn, sort_keys=True) for turn in row.get("source_turn_ids", []) if turn is not None}
        if (row["hash"] in old_ids or prefix in old_prefixes or turns & old_turns
                or (row["source_file"], row["source_row"]) in old_sources):
            raise ValueError("prospective reference overlaps previously screened source")


def load_references(path, min_turns, max_turns):
    """Verify frozen v3 labels, source windows, and exclusions before model loading."""
    path = Path(path)
    manifest = json.loads(Path(str(path) + ".manifest.json").read_bytes())
    selection.verify_manifest(manifest)
    if manifest.get("schema_version") != 2 or manifest.get("references_sha256") != selection.file_checksum(path):
        raise ValueError("references do not match their v3 provenance manifest; legacy references need re-review")
    if manifest.get("policy_sha256") != selection.checksum(selection.POLICY.encode()):
        raise ValueError("reference policy changed; re-review references before evaluation")
    if manifest.get("taxonomy_sha256") != selection.taxonomy_metadata()["sha256"]:
        raise ValueError("reference taxonomy changed; re-review references before evaluation")
    selection.immutable_revision(manifest["dataset"]["revision"])
    retained = manifest["retained_window"]
    if retained != {"min_turns": min_turns, "max_turns": max_turns}:
        raise ValueError("reference retained window must match evaluation limits")
    freeze = manifest.get("freeze", {})
    if (freeze.get("stage") != "before_classifier_predictions"
            or any(freeze.get(field) is not False for field in
                   ("classifier_predictions_seen", "projection_outcomes_seen", "prompt_development_examples"))):
        raise ValueError("references must be frozen before predictions without outcome or prompt-development leakage")
    cases = selection.read_jsonl(path)
    if not cases:
        raise ValueError("references must not be empty")
    for case in cases:
        row = case["conversation"]
        selection.validate_candidate(row, min_turns, max_turns)
        if ("reference_subtopic" not in case
                or not _validate_label(case.get("reference_domain"), case.get("reference_subtopic"))
                or case.get("reference_status") not in REFERENCE_STATUSES):
            raise ValueError("invalid reference domain/subtopic pair or adjudication status")
        if case.get("split") not in SPLITS or not isinstance(case.get("reference_rationale"), str) or not case["reference_rationale"].strip():
            raise ValueError("reference needs a split and explicit rationale")
        if (not isinstance(row.get("source_file"), str) or not row["source_file"]
                or type(row.get("source_row")) is not int or row["source_row"] < 0
                or type(row.get("source_turns")) is not int
                or len(row["users"]) != min(row["source_turns"], max_turns)):
            raise ValueError("reference must retain every original initial user turn up to the maximum")
        if case["reference_domain"] not in selection.DOMAINS:
            kind = case.get("negative_kind")
            if kind not in coverage.NEGATIVE_KINDS or case["reference_domain"] != ("other" if kind == "non_english" else kind):
                raise ValueError("negative reference needs a matching negative_kind")
    _, duplicates = selection.deduplicate([case["conversation"] for case in cases])
    if duplicates:
        raise ValueError("references contain duplicate or overlapping conversations")
    audit_ids = set(manifest.get("previous_audit_ids", []))
    if any(case["split"] in {"heldout", "coverage"} and case["conversation"]["hash"] in audit_ids for case in cases):
        raise ValueError("heldout or coverage references overlap the previous audit")
    if manifest.get("counts") != dict(Counter(case["split"] for case in cases)):
        raise ValueError("reference split counts do not match manifest")
    _source_isolation(cases, manifest, path.parent)
    return cases, manifest


def _rates(confusion, labels):
    rates = {}
    for label in labels:
        reference_n = sum(confusion[label].values())
        predicted_n = sum(row.get(label, 0) for row in confusion.values())
        matched = confusion[label].get(label, 0)
        rates[label] = {"reference_n": reference_n, "predicted_n": predicted_n, "agreement_n": matched,
                        "precision_wrt_reference": matched / predicted_n if predicted_n else None,
                        "recall_wrt_reference": matched / reference_n if reference_n else None}
    return rates


def _topic_key(domain, topic):
    return f"{domain}/{topic}" if topic is not None else f"{domain}/unassigned"


def summarize(cases, predictions):
    """Domain and subtopic agreement; invalid/unassigned predictions stay in denominators."""
    expected = {case["conversation"]["hash"]: case for case in cases}
    actual = {row["hash"]: row for row in predictions}
    if len(expected) != len(cases) or len(actual) != len(predictions) or set(expected) != set(actual):
        raise ValueError("predictions must cover every reference exactly once")
    labels = [*selection.DOMAINS, "mixed", "roleplay", "other"]
    confusion = {label: dict.fromkeys([*labels, *INVALID_STATUSES], 0) for label in labels}
    topic_labels = [_topic_key(domain, topic) for domain in selection.DOMAINS for topic in selection.SUBTOPICS[domain]]
    topic_columns = [*topic_labels, *[_topic_key(domain, None) for domain in labels], *INVALID_STATUSES]
    topic_reference_labels = [*topic_labels, *[_topic_key(domain, None) for domain in ("mixed", "roleplay", "other")]]
    topic_confusion = {topic: dict.fromkeys(topic_columns, 0) for topic in topic_reference_labels}
    statuses, normalizations = Counter(), Counter()
    correct = topic_correct = topic_n = joint_correct = joint_n = unassigned_predictions = unassigned_references = 0
    disagreements = []
    for key, case in expected.items():
        prediction = actual[key]
        status = prediction["status"]
        if status not in {"classified", *INVALID_STATUSES}:
            raise ValueError(f"unknown prediction status: {status}")
        domain = prediction["domain"] if status == "classified" else status
        topic = prediction.get("subtopic") if status == "classified" else None
        if status == "classified" and ("subtopic" not in prediction or not _validate_label(domain, topic)):
            raise ValueError("prediction has invalid domain/subtopic pair")
        if not _validate_label(case["reference_domain"], case.get("reference_subtopic")):
            raise ValueError("invalid reference domain/subtopic pair")
        expected_domain, expected_topic = case["reference_domain"], case.get("reference_subtopic")
        confusion[expected_domain][domain] += 1
        statuses[status] += 1
        normalizations[prediction.get("normalization") or "none"] += 1
        matched = status == "classified" and domain == expected_domain
        correct += matched
        unassigned_predictions += status == "classified" and domain in selection.DOMAINS and topic is None
        unassigned_references += expected_domain in selection.DOMAINS and expected_topic is None
        topic_match = matched and topic == expected_topic
        scorable_pair = expected_domain not in selection.DOMAINS or expected_topic is not None
        if scorable_pair:
            joint_n += 1
            joint_correct += topic_match
            predicted_topic = _topic_key(domain, topic) if status == "classified" else status
            topic_confusion[_topic_key(expected_domain, expected_topic)][predicted_topic] += 1
        if expected_domain in selection.DOMAINS and expected_topic is not None:
            topic_n += 1
            topic_correct += topic_match
        if not matched or scorable_pair and not topic_match:
            disagreements.append({"hash": key, "split": case["split"], "reference_domain": expected_domain,
                                  "reference_subtopic": expected_topic, "domain_disagreement": not matched,
                                  "subtopic_disagreement": not topic_match if expected_topic is not None else None,
                                  "reference_rationale": case["reference_rationale"], "reference_status": case["reference_status"], "prediction": prediction})
    metrics = {
        "n": len(cases), "agreement_n": correct, "agreement": correct / len(cases) if cases else None,
        "confusion_reference_by_prediction": confusion, "per_class": _rates(confusion, labels),
        "subtopic": {"n": topic_n, "agreement_n": topic_correct, "agreement": topic_correct / topic_n if topic_n else None,
                     "confusion_reference_by_prediction": topic_confusion, "per_class": _rates(topic_confusion, topic_labels),
                     "denominator": "References with an assigned target subtopic; includes wrong-domain, invalid, and unassigned predictions as disagreements."},
        "joint_domain_subtopic": {"n": joint_n, "agreement_n": joint_correct, "agreement": joint_correct / joint_n if joint_n else None},
        "unassigned_prediction_subtopic_n": unassigned_predictions,
        "unassigned_reference_subtopic_n": unassigned_references,
        "status_counts": dict(statuses), "invalid_output_counts": {status: statuses[status] for status in INVALID_STATUSES},
        "truncated_output_n": sum(row.get("truncated") is True for row in predictions),
        "normalization_counts": dict(normalizations),
        "reference_status_counts": dict(Counter(case["reference_status"] for case in cases)),
    }
    return metrics, sorted(disagreements, key=lambda row: row["hash"])


def run(args, model_loader=judge.load_judge):
    revision = selection.immutable_revision(args.model_revision)
    cases, reference_manifest = load_references(args.references, args.min_turns, args.max_turns)
    output = Path(args.run_dir)
    paths = {"predictions_sha256": output / "predictions.jsonl", "disagreements_sha256": output / "disagreements.jsonl", "metrics_sha256": output / "metrics.json"}
    identity = {
        "references_sha256": selection.file_checksum(args.references),
        "reference_manifest_sha256": selection.file_checksum(str(args.references) + ".manifest.json"),
        "model": MODEL, "revision": revision, "policy_version": selection.POLICY_VERSION,
        "structured_outputs": {"json": selection.LABEL_SCHEMA},
        "policy_sha256": selection.checksum(selection.POLICY.encode()),
        "taxonomy": selection.taxonomy_metadata(),
        "configuration": {key: getattr(args, key) for key in ("min_turns", "max_turns", "context_limit", "output_tokens")},
        "runtime": judge.runtime_versions(),
        "code": selection.code_hashes(__file__, judge.__file__, selection.__file__, coverage.__file__),
    }
    manifest_path = output / "evaluation.manifest.json"
    with selection.selection_lock(output / ".evaluation.lock"):
        cached = selection.load_completed(manifest_path, identity, paths)
        if cached:
            print(paths["metrics_sha256"].read_text())
            return cached
        tokenizer, complete = model_loader(MODEL, revision, args.context_limit, args.output_tokens)
        predictions = judge.classify([case["conversation"] for case in cases], tokenizer, complete, args.context_limit, args.output_tokens)
        overall, disagreements = summarize(cases, predictions)
        metrics = {
            "interpretation": "Agreement with explicitly labeled references; AI-reviewed provisional references are not human ground truth. Deliberately enriched sampling is not a population error-rate estimate. No automatic study-quality pass threshold.",
            "retained_window": reference_manifest["retained_window"],
            "requested_max_turns": args.max_turns,
            "population_limitations": reference_manifest.get("population_limitations", []),
            "overall": overall,
            "reference_coverage": coverage.coverage_report(cases),
            "readiness": coverage.readiness_report(cases, references_frozen=True, source_isolation_verified=True, classifier_execution_complete=True),
            "by_split": {},
        }
        for split in sorted(SPLITS):
            subset = [case for case in cases if case["split"] == split]
            ids = {case["conversation"]["hash"] for case in subset}
            metrics["by_split"][split] = summarize(subset, [row for row in predictions if row["hash"] in ids])[0]
        artifacts = {"predictions_sha256": selection.jsonl_bytes(predictions), "disagreements_sha256": selection.jsonl_bytes(disagreements), "metrics_sha256": selection.json_bytes(metrics)}
        manifest = selection.seal_manifest({"schema_version": 1, "identity": identity, "reference_provenance": reference_manifest, "review_status": "requires_disagreement_review", "counts": {"references": len(cases), "disagreements": len(disagreements)}, **{key: selection.checksum(raw) for key, raw in artifacts.items()}})
        for key, raw in artifacts.items():
            selection.atomic_write(paths[key], raw)
        selection.atomic_write(manifest_path, selection.json_bytes(manifest))
        print(json.dumps({"agreement": overall["agreement"], "n": overall["n"], "invalid_output_counts": overall["invalid_output_counts"], "metrics": str(paths["metrics_sha256"]), "review_status": manifest["review_status"]}, indent=2))
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("references")
    parser.add_argument("run_dir")
    parser.add_argument("--model-revision", required=True, help="immutable 40-character revision used by the production classifier")
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--output-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    if not 1 <= args.min_turns <= args.max_turns or not 0 < args.output_tokens < args.context_limit:
        parser.error("require 1 <= min-turns <= max-turns and 0 < output-tokens < context-limit")
    run(args)


if __name__ == "__main__":
    main()
