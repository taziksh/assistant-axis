"""CPU contracts for reference provenance and metrics; no LLM quality claims."""

import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wild.selection import evaluate, helpers, judge

SHA = "a" * 40


def reference(key, domain="coding", split="heldout", topic="default", turns=3):
    topic = helpers.SUBTOPICS[domain][0] if topic == "default" and domain in helpers.DOMAINS else None if topic == "default" else topic
    case = {"conversation": {"hash": key, "users": [f"{key} user {i}" for i in range(turns)], "gpt": [f"{key} reply {i}" for i in range(turns)], "source_file": "data/fresh.parquet", "source_row": sum((i + 1) * ord(c) for i, c in enumerate(key)), "source_turns": turns, "source_turn_ids": [f"{key}-{i}" for i in range(2 * turns)]}, "reference_domain": domain, "reference_subtopic": topic, "reference_rationale": "Provisional fixture reference.", "reference_status": "ai_reviewed_provisional", "split": split}
    if domain not in helpers.DOMAINS:
        case["negative_kind"] = domain
    return case


def write_references(path, cases, exclusions=None, source=None, **changes):
    raw = helpers.jsonl_bytes(cases)
    path.write_bytes(raw)
    source_path = path.parent / "source.jsonl"
    source_path.write_bytes(helpers.jsonl_bytes(source if source is not None else [case["conversation"] for case in cases]))
    exclusion_path = path.parent / "exclusions.json"
    exclusion_path.write_bytes(helpers.json_bytes(exclusions if exclusions is not None else {"conversation_hashes": [], "first3_exchange_sha256": [], "source_turn_id_json": [], "source_identities": []}))
    manifest = {"schema_version": 2, "dataset": {"repo_id": "fixture/data", "revision": SHA}, "references_sha256": helpers.checksum(raw), "policy_sha256": helpers.checksum(helpers.POLICY.encode()), "taxonomy_sha256": helpers.taxonomy_metadata()["sha256"], "retained_window": {"min_turns": 3, "max_turns": 15}, "previous_audit_ids": ["old"], "counts": dict(Counter(case["split"] for case in cases)), "population_limitations": ["CPU fixture; not empirical validation."], "freeze": {"stage": "before_classifier_predictions", "classifier_predictions_seen": False, "projection_outcomes_seen": False, "prompt_development_examples": False}, "source_artifact": {"path": source_path.name, "sha256": helpers.file_checksum(source_path)}, "exclusion_artifact": {"path": exclusion_path.name, "sha256": helpers.file_checksum(exclusion_path)}, **changes}
    Path(str(path) + ".manifest.json").write_bytes(helpers.json_bytes(helpers.seal_manifest(manifest)))


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(json.dumps(messages).encode())


def args(path, output):
    return argparse.Namespace(references=str(path), run_dir=str(output), model_revision=SHA, min_turns=3, max_turns=15, context_limit=16384, output_tokens=256)


def prediction(key, domain=None, status="classified", **extra):
    return {"hash": key, "domain": domain, "subtopic": helpers.SUBTOPICS[domain][0] if domain in helpers.DOMAINS else None, "status": status, "raw": "raw fixture", **extra}


def test_metrics_count_invalid_outputs_in_denominators_and_keep_empty_classes_null():
    cases = [reference("a"), reference("b"), reference("c", "writing"), reference("d", "therapy")]
    predictions = [prediction("a", "coding"), prediction("b", "writing"), prediction("c", status="unparsed", truncated=True), prediction("d", status="too_long")]
    metrics, disagreements = evaluate.summarize(cases, predictions)
    assert metrics["agreement_n"] == 1 and metrics["agreement"] == 0.25
    assert metrics["invalid_output_counts"] == {"unparsed": 1, "too_long": 1}
    assert metrics["truncated_output_n"] == 1
    assert metrics["per_class"]["coding"]["recall_wrt_reference"] == 0.5
    assert metrics["per_class"]["writing"]["precision_wrt_reference"] == 0
    assert metrics["per_class"]["ai"]["recall_wrt_reference"] is None
    assert {row["hash"] for row in disagreements} == {"b", "c", "d"}
    assert metrics["confusion_reference_by_prediction"]["therapy"]["too_long"] == 1


@pytest.mark.parametrize("predictions", [[], [prediction("a", "coding"), prediction("a", "coding")], [prediction("wrong", "coding")]])
def test_metrics_reject_missing_duplicate_or_foreign_predictions(predictions):
    with pytest.raises(ValueError, match="exactly once"):
        evaluate.summarize([reference("a")], predictions)


def test_evaluation_uses_production_classifier_and_separates_splits(tmp_path, monkeypatch):
    path, output = tmp_path / "references.jsonl", tmp_path / "run"
    cases = [reference("old", split="regression"), reference("new", "writing"), reference("targeted", "ai", split="coverage")]
    write_references(path, cases)
    called = []
    actual_classify = judge.classify
    def classify(*parameters):
        called.append(parameters[0])
        return actual_classify(*parameters)
    monkeypatch.setattr(judge, "classify", classify)
    def loader(model, revision, context, tokens):
        assert revision == SHA and context == 16384 and tokens == 256
        return Tokenizer(), lambda prompts: [{"text": '{"domain":"coding","subtopic":"implementation","rationale":"Fixture only."}', "finish_reason": "stop", "output_tokens": 12} for _ in prompts]
    manifest = evaluate.run(args(path, output), model_loader=loader)
    assert called == [[case["conversation"] for case in cases]]
    metrics = json.loads((output / "metrics.json").read_bytes())
    assert metrics["overall"]["agreement"] == 1 / 3
    assert metrics["by_split"]["regression"]["agreement"] == 1
    assert metrics["by_split"]["heldout"]["agreement"] == 0
    assert metrics["by_split"]["coverage"]["n"] == 1 and metrics["by_split"]["coverage"]["agreement"] == 0
    assert "not human ground truth" in metrics["interpretation"]
    assert metrics["retained_window"]["max_turns"] == 15 and metrics["requested_max_turns"] == 15
    assert manifest["review_status"] == "requires_disagreement_review"
    assert metrics["readiness"]["classifier_execution_complete"] is True
    assert metrics["readiness"]["classifier_validation_complete"] is False
    assert helpers.read_jsonl(output / "predictions.jsonl")[0]["raw"]
    assert evaluate.run(args(path, output), model_loader=lambda *_: pytest.fail("cache loaded model")) == manifest


@pytest.mark.parametrize("failure", ["policy", "data", "audit_overlap", "duplicate", "bad_status", "bad_label", "window", "floating_revision", "taxonomy", "topic_pair", "missing_topic", "source_window", "unfrozen", "legacy_schema"])
def test_invalid_references_fail_before_model_loading(tmp_path, failure):
    path = tmp_path / "references.jsonl"
    cases, changes = [reference("new")], {}
    if failure == "policy":
        changes["policy_sha256"] = "0" * 64
    elif failure == "audit_overlap":
        cases[0]["conversation"]["hash"] = "old"
    elif failure == "duplicate":
        cases.append(copy.deepcopy(cases[0]))
    elif failure == "bad_status":
        cases[0]["reference_status"] = "ground_truth"
    elif failure == "bad_label":
        cases[0]["reference_domain"] = "unknown"
    elif failure == "window":
        changes["retained_window"] = {"min_turns": 3, "max_turns": 16}
    elif failure == "taxonomy":
        changes["taxonomy_sha256"] = "0" * 64
    elif failure == "topic_pair":
        cases[0]["reference_subtopic"] = "relationships"
    elif failure == "missing_topic":
        del cases[0]["reference_subtopic"]
    elif failure == "source_window":
        cases[0]["conversation"]["source_turns"] = 4
    elif failure == "unfrozen":
        changes["freeze"] = {"stage": "after_predictions"}
    elif failure == "legacy_schema":
        changes["schema_version"] = 1
    write_references(path, cases, **changes)
    if failure == "data":
        path.write_bytes(path.read_bytes() + b"\n")
    config = args(path, tmp_path / "run")
    if failure == "floating_revision":
        config.model_revision = "main"
    with pytest.raises(ValueError):
        evaluate.run(config, model_loader=lambda *_: pytest.fail("invalid input loaded model"))


def test_changed_outputs_or_configuration_cannot_reuse_evaluation(tmp_path):
    path, output = tmp_path / "references.jsonl", tmp_path / "run"
    write_references(path, [reference("new")])
    loader = lambda *_: (Tokenizer(), lambda prompts: ['{"domain":"coding","subtopic":"implementation","rationale":"Fixture only."}'] * len(prompts))
    config = args(path, output)
    evaluate.run(config, model_loader=loader)
    config.output_tokens = 257
    with pytest.raises(ValueError, match="settings changed"):
        evaluate.run(config, model_loader=lambda *_: pytest.fail("cache loaded model"))
    config.output_tokens = 256
    (output / "metrics.json").write_text("{}\n")
    with pytest.raises(ValueError, match="modified selection output"):
        evaluate.run(config, model_loader=lambda *_: pytest.fail("modified output loaded model"))


def test_subtopic_denominators_include_invalid_unassigned_wrong_domain_and_negative_false_positives():
    cases = [reference("ok"), reference("missing"), reference("invalid"), reference("wrong_domain"),
             reference("wrong_topic"), reference("unknown", topic=None), reference("negative", "other")]
    predictions = [prediction("ok", "coding"), prediction("missing", "coding", subtopic=None),
                   prediction("invalid", status="unparsed"), prediction("wrong_domain", "writing"),
                   prediction("wrong_topic", "coding", subtopic="debugging_testing"),
                   prediction("unknown", "coding"), prediction("negative", "coding")]
    metrics, disagreements = evaluate.summarize(cases, predictions)
    assert metrics["agreement_n"] == 4
    assert metrics["subtopic"]["n"] == 5 and metrics["subtopic"]["agreement"] == 1 / 5
    assert metrics["joint_domain_subtopic"] == {"n": 6, "agreement_n": 1, "agreement": 1 / 6}
    assert metrics["unassigned_prediction_subtopic_n"] == 1
    assert metrics["unassigned_reference_subtopic_n"] == 1
    rates = metrics["subtopic"]["per_class"]["coding/implementation"]
    assert rates["recall_wrt_reference"] == 1 / 5
    assert rates["precision_wrt_reference"] == 1 / 2
    assert {row["hash"] for row in disagreements} == {"missing", "invalid", "wrong_domain", "wrong_topic", "negative"}


def test_metrics_reject_classified_domain_subtopic_mismatch():
    with pytest.raises(ValueError, match="invalid domain/subtopic pair"):
        evaluate.summarize([reference("new")], [prediction("new", "coding", subtopic="relationships")])


@pytest.mark.parametrize("identity", ["id", "prefix", "turn", "source"])
def test_prospective_reference_source_leakage_fails_before_model_loading(tmp_path, identity):
    case = reference("new")
    row = case["conversation"]
    exclusions = {"conversation_hashes": [], "first3_exchange_sha256": [], "source_turn_id_json": [], "source_identities": []}
    if identity == "id":
        exclusions["conversation_hashes"] = [row["hash"]]
    elif identity == "prefix":
        exclusions["first3_exchange_sha256"] = [helpers.checksum(helpers.json_bytes(list(zip(row["users"], row["gpt"]))))]
    elif identity == "turn":
        exclusions["source_turn_id_json"] = [json.dumps(row["source_turn_ids"][0], sort_keys=True)]
    elif identity == "source":
        exclusions["source_identities"] = [{"source_file": row["source_file"], "source_row": row["source_row"], "hash": "old-different-id"}]
    path = tmp_path / "references.jsonl"
    write_references(path, [case], exclusions=exclusions)
    with pytest.raises(ValueError, match="previously screened"):
        evaluate.run(args(path, tmp_path / "run"), model_loader=lambda *_: pytest.fail("leakage loaded model"))


def test_reference_text_must_equal_original_source_and_not_drop_later_turns(tmp_path):
    case = reference("new", turns=15)
    source = copy.deepcopy(case["conversation"])
    case["conversation"]["users"][-1] = "Synthetic replacement"
    path = tmp_path / "references.jsonl"
    write_references(path, [case], source=[source])
    with pytest.raises(ValueError, match="original source window"):
        evaluate.load_references(path, 3, 15)


def test_fifteen_turn_references_reach_production_classifier_intact(tmp_path):
    case = reference("new", "ai", turns=15)
    path = tmp_path / "references.jsonl"
    write_references(path, [case])
    seen = []
    class InspectTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            seen.append(json.loads(messages[-1]["content"])["user_messages"])
            return super().apply_chat_template(messages, **kwargs)
    loader = lambda *_: (InspectTokenizer(), lambda prompts: ['{"domain":"ai","subtopic":"consciousness_experience","rationale":"Fixture only."}'] * len(prompts))
    evaluate.run(args(path, tmp_path / "run"), model_loader=loader)
    assert seen == [case["conversation"]["users"]]
