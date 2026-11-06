"""Offline quota/schema contracts; synthetic fixtures do not establish source yield."""

from collections import Counter
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wild.selection import helpers


def pool(size=25):
    rows, labels = [], []
    for domain, topics in helpers.SUBTOPICS.items():
        for topic in topics:
            for i in range(size):
                key = f"{domain}/{topic}/{i:02}"
                rows.append({"hash": key, "users": [key], "gpt": ["Fixture reply."]})
                labels.append({"hash": key, **helpers.parse_label(json.dumps({
                    "domain": domain, "subtopic": topic, "rationale": "Synthetic fixture only.",
                }))})
    return rows, labels


def current_policy():
    return {"policy_version": helpers.POLICY_VERSION, "policy_sha256": helpers.checksum(helpers.POLICY.encode())}


@pytest.mark.parametrize("domain,topic", [
    ("coding", "relationships"), ("therapy", "implementation"),
    ("ai", "unknown"), ("writing", "Drafting"), ("coding", []),
    ("coding", 1), ("coding", {}), ("roleplay", "creative_fiction"),
    ("mixed", "implementation"), ("other", "consciousness_experience"),
])
def test_invalid_domain_subtopic_pairs_are_explicit_exclusions(domain, topic):
    raw = json.dumps({"domain": domain, "subtopic": topic, "rationale": "Fixture."})
    label = helpers.parse_label(raw)
    assert label["status"] == "unparsed"
    assert label["domain"] is None and label["subtopic"] is None
    assert label["raw"] == raw


def test_old_domain_only_json_never_receives_an_invented_subtopic():
    label = helpers.parse_label('{"domain":"coding","rationale":"Code request."}')
    assert label["status"] == "unparsed" and label["subtopic"] is None


@pytest.mark.parametrize("domain", sorted(helpers.LABELS))
def test_explicit_unresolved_or_excluded_subtopic_can_be_null(domain):
    label = helpers.parse_label(json.dumps({"domain": domain, "subtopic": None, "rationale": "Unresolved or excluded."}))
    assert label["status"] == "classified" and label["subtopic"] is None


@pytest.mark.parametrize("quota", [10, 100])
def test_equal_seeded_quotas_cover_all_twenty_strata_and_preserve_metadata(quota):
    rows, labels = pool()
    selected, audit, counts = helpers.select_domains(rows, labels, quota, 7)
    assert len(selected) == quota * 4
    expected = {(domain, topic): quota // 5 for domain, topics in helpers.SUBTOPICS.items() for topic in topics}
    assert Counter((row["domain"], row["subtopic"]) for row in selected) == expected
    assert all(count == 0 for count in counts["shortfalls"].values())
    assert counts["available_counts"] == dict.fromkeys(helpers.DOMAINS, 125)
    assert {row["hash"] for row in audit} == {row["hash"] for row in rows}
    assert all(counts["selected_subtopic_counts"][domain][topic] == quota // 5 for domain, topic in expected)
    assert helpers.select_domains(list(reversed(rows)), list(reversed(labels)), quota, 7) == (selected, audit, counts)
    assert helpers.select_domains(rows, labels, quota, 8)[0] != selected
    assert [json.loads(line) for line in helpers.jsonl_bytes(selected).splitlines()] == selected


def test_missing_buckets_redistribute_to_named_subtopics_before_using_broad_cases():
    rows, labels = pool(30)
    missing = "ai/moral_status_values/"
    rows = [row for row in rows if not row["hash"].startswith(missing)]
    labels = [label for label in labels if not label["hash"].startswith(missing)]
    unassigned = {"hash": "ai/unresolved", "users": ["Fixture."], "gpt": ["Fixture reply."]}
    rows.append(unassigned)
    labels.append({"hash": unassigned["hash"], **helpers.parse_label(json.dumps({
        "domain": "ai", "subtopic": None, "rationale": "No supported subtopic.",
    }))})
    selected, audit, counts = helpers.select_domains(rows, labels, 100, 0)
    assert counts["selected_counts"]["ai"] == 100
    assert counts["available_counts"]["ai"] == 121
    assert counts["shortfalls"]["ai"] == 0
    assert counts["available_subtopic_counts"]["ai"]["moral_status_values"] == 0
    assert counts["selected_subtopic_counts"]["ai"]["moral_status_values"] == 0
    assert counts["subtopic_shortfalls"]["ai"]["moral_status_values"] == 20
    assert next(row for row in audit if row["hash"] == "ai/unresolved")["outcome"] == "quota_filled"
    assert counts["broad_available_counts"]["ai"] == 1
    assert counts["broad_selected_counts"]["ai"] == 0
    assert sum(counts["subtopic_redistribution"]["ai"].values()) == 20
    assert set(value for topic, value in counts["selected_subtopic_counts"]["ai"].items() if topic != "moral_status_values") == {25}
    assert all(row["hash"] != "ai/unresolved" for row in selected)


def test_another_buckets_size_cannot_reorder_existing_selection():
    rows, labels = pool()
    full = helpers.select_domains(rows, labels, 100, 17)[0]
    rows = [row for row in rows if not row["hash"].startswith("ai/")]
    labels = [row for row in labels if not row["hash"].startswith("ai/")]
    smaller = helpers.select_domains(rows, labels, 100, 17)[0]
    assert smaller == [row for row in full if row["domain"] != "ai"]


def test_broad_fallback_preserves_domain_quality_and_never_crosses_domains():
    rows, labels = pool(3)
    for i in range(90):
        key = f"ai/broad/{i}"
        rows.append({"hash": key, "users": [key], "gpt": ["Fixture reply."]})
        labels.append({"hash": key, **helpers.parse_label(json.dumps({
            "domain": "ai", "subtopic": None, "rationale": "Valid AI focus; unresolved subtopic.",
        }))})
    for excluded in ("other", "mixed", "roleplay"):
        rows.append({"hash": excluded, "users": [excluded], "gpt": ["Fixture."]})
        labels.append({"hash": excluded, **helpers.parse_label(json.dumps({
            "domain": excluded, "subtopic": None, "rationale": "Excluded by domain rules.",
        }))})
    selected, audit, counts = helpers.select_domains(rows, labels, 100, 9)
    assert counts["selected_counts"] == {"coding": 15, "writing": 15, "therapy": 15, "ai": 100}
    assert counts["broad_selected_counts"] == {"coding": 0, "writing": 0, "therapy": 0, "ai": 85}
    assert counts["shortfalls"] == {"coding": 85, "writing": 85, "therapy": 85, "ai": 0}
    assert all(row["hash"] not in {"other", "mixed", "roleplay"} for row in selected)
    assert sum(row["outcome"] == "selected" and row["selection_group"] == "broad_unassigned" for row in audit) == 85
    assert helpers.select_domains(list(reversed(rows)), labels, 100, 9) == (selected, audit, counts)


@pytest.mark.parametrize("quota", [1, 3, 99, 101])
def test_arbitrary_positive_domain_quotas_balance_with_at_most_one_extra(quota):
    rows, labels = pool()
    selected, _, counts = helpers.select_domains(rows, labels, quota, 0)
    assert len(selected) == 4 * quota
    for domain in helpers.DOMAINS:
        achieved = list(counts["selected_subtopic_counts"][domain].values())
        assert max(achieved) - min(achieved) <= 1
        assert counts["selected_subtopic_counts"][domain] == counts["target_subtopic_counts"][domain]


@pytest.mark.parametrize("quota", [0, -5, True, 100.0])
def test_invalid_quotas_fail_before_selection(quota):
    with pytest.raises(ValueError, match="positive integer"):
        helpers.select_domains([], [], quota, 0)


def test_measurement_gate_checks_frozen_subtopics_and_actual_records():
    rows, labels = pool()
    selected, audit, counts = helpers.select_domains(rows, labels, 100, 3)
    manifest = helpers.seal_manifest({
        "identity": current_policy(), "judge": current_policy(),
        "selection": {"per_domain": 100, "target_per_subtopic": 20, "stratification": "best_effort_subtopics", "seed": 3},
        "taxonomy": helpers.taxonomy_metadata(), "selection_strategy": helpers.selection_strategy_metadata(), "counts": counts,
    })
    assert helpers.verify_selection_quotas(manifest, selected, rows, labels, audit) == 100
    # Same total per domain, but one topic has 21 and another has 19.
    changed = copy.deepcopy(selected)
    changed[0]["subtopic"] = next(topic for topic in helpers.SUBTOPICS[changed[0]["domain"]] if topic != changed[0]["subtopic"])
    with pytest.raises(ValueError, match="records"):
        helpers.verify_selection_quotas(manifest, changed, rows, labels, audit)
    missing = copy.deepcopy(manifest)
    missing["counts"]["selected_subtopic_counts"]["ai"]["moral_status_values"] = 19
    with pytest.raises(ValueError, match="audit"):
        helpers.verify_selection_quotas(helpers.seal_manifest(missing), selected, rows, labels, audit)
    stale = copy.deepcopy(manifest)
    stale["taxonomy"]["version"] = "old-taxonomy"
    with pytest.raises(ValueError, match="taxonomy"):
        helpers.verify_selection_quotas(helpers.seal_manifest(stale), selected, rows, labels, audit)
    changed_audit = copy.deepcopy(audit)
    changed_audit[0]["selection_group"] = "broad_unassigned"
    with pytest.raises(ValueError, match="audit"):
        helpers.verify_selection_quotas(manifest, selected, rows, labels, changed_audit)


def test_measurement_accepts_reproducible_redistribution_without_twenty_per_bucket():
    rows, labels = pool()
    rows = [row for row in rows if not row["hash"].startswith("ai/moral_status_values/")]
    labels = [row for row in labels if not row["hash"].startswith("ai/moral_status_values/")]
    selected, audit, counts = helpers.select_domains(rows, labels, 100, 3)
    manifest = helpers.seal_manifest({
        "identity": current_policy(), "judge": current_policy(),
        "selection": {"per_domain": 100, "target_per_subtopic": 20, "stratification": "best_effort_subtopics", "seed": 3},
        "taxonomy": helpers.taxonomy_metadata(), "selection_strategy": helpers.selection_strategy_metadata(), "counts": counts,
    })
    assert counts["selected_subtopic_counts"]["ai"]["moral_status_values"] == 0
    assert counts["selected_counts"]["ai"] == 100
    assert helpers.verify_selection_quotas(manifest, selected, rows, labels, audit) == 100


@pytest.mark.parametrize("section", ["identity", "judge"])
@pytest.mark.parametrize("field", ["policy_version", "policy_sha256"])
def test_measurement_rejects_stale_policy_even_when_selection_replays(section, field):
    rows, labels = pool(1)
    selected, audit, counts = helpers.select_domains(rows, labels, 5, 3)
    manifest = {
        "identity": current_policy(), "judge": current_policy(),
        "selection": {"per_domain": 5, "target_per_subtopic": 1, "stratification": "best_effort_subtopics", "seed": 3},
        "taxonomy": helpers.taxonomy_metadata(), "selection_strategy": helpers.selection_strategy_metadata(), "counts": counts,
    }
    manifest[section][field] = "older-policy"
    with pytest.raises(ValueError, match="current domain policy"):
        helpers.verify_selection_quotas(helpers.seal_manifest(manifest), selected, rows, labels, audit)


def test_selection_rejects_handwritten_labels_without_valid_subtopic_pairs():
    rows, labels = pool(1)
    labels[0]["subtopic"] = "relationships"
    with pytest.raises(ValueError, match="domain/subtopic"):
        helpers.select_domains(rows, labels, 5, 0)
