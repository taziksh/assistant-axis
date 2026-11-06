"""Coverage and readiness are not established by empty classes or CPU fixtures."""

from wild.selection import coverage, helpers


def case(key, domain, topic=None, turns=3, split="heldout", negative_kind=None):
    return {"conversation": {"hash": key, "users": [f"Fixture user {i}" for i in range(turns)]},
            "split": split, "reference_domain": domain, "reference_subtopic": topic,
            "reference_status": "ai_reviewed_provisional", "negative_kind": negative_kind}


def complete_fixture():
    cases = []
    for domain, topics in helpers.SUBTOPICS.items():
        for topic in topics:
            for number in range(5):
                cases.append(case(f"{domain}-{topic}-{number}", domain, topic,
                                  turns=13 + number % 3 if domain in coverage.RARE_DOMAINS else 3))
    for kind in coverage.NEGATIVE_KINDS:
        for number in range(5):
            cases.append(case(f"{kind}-{number}", "other" if kind == "non_english" else kind, negative_kind=kind))
    return cases


def test_empty_references_are_not_coverage_complete():
    report = coverage.coverage_report([])
    assert report["coverage_complete"] is False
    assert report["positive_n"] == report["negative_n"] == 0
    assert len(report["shortfalls"]) == 10
    assert len(report["advisory_shortfalls"]) == 20
    assert report["long_rare_positive_counts"] == {"therapy": 0, "ai": 0}


def test_coverage_targets_are_separate_from_classifier_execution_and_human_adjudication():
    cases = complete_fixture()
    report = coverage.readiness_report(cases, code_checks_complete=True)
    assert report["reference_coverage_complete"] is True
    assert report["code_checks_complete"] is True
    assert report["ready_for_classifier_validation"] is False
    assert report["classifier_validation_complete"] is False
    assert report["study_quality_approval"] == "not_assessed"
    ready = coverage.readiness_report(cases, references_frozen=True, source_isolation_verified=True, code_checks_complete=True)
    assert ready["ready_for_classifier_validation"] is True
    assert ready["coverage"]["reference_status_counts"] == {"ai_reviewed_provisional": 120}


def test_execution_with_incomplete_coverage_is_diagnostic_not_completed_validation():
    report = coverage.readiness_report([], references_frozen=True, source_isolation_verified=True,
                                       classifier_execution_complete=True, code_checks_complete=True)
    assert report["ready_for_classifier_diagnostics"] is True
    assert report["ready_for_classifier_validation"] is False
    assert report["classifier_execution_complete"] is True
    assert report["classifier_validation_complete"] is False
    full = coverage.readiness_report(complete_fixture(), references_frozen=True, source_isolation_verified=True,
                                     classifier_execution_complete=True)
    assert full["ready_for_classifier_validation"] is False
    assert full["classifier_validation_complete"] is False
    assert full["disagreement_review_complete"] is False
    no_checks = coverage.readiness_report(complete_fixture(), references_frozen=True, source_isolation_verified=True,
                                          classifier_execution_complete=True, disagreement_review_complete=True)
    assert no_checks["classifier_validation_complete"] is False


def test_old_short_windows_and_nonheldout_examples_do_not_fill_rare_coverage():
    cases = complete_fixture()
    for item in cases:
        if item["reference_domain"] in coverage.RARE_DOMAINS:
            item["conversation"]["users"] = item["conversation"]["users"][:12]
    for domain in coverage.RARE_DOMAINS:
        for number in range(10):
            cases.append(case(f"targeted-{domain}-{number}", domain, helpers.SUBTOPICS[domain][0], turns=15, split="coverage"))
    report = coverage.coverage_report(cases)
    assert report["positive_n"] == 100
    assert report["coverage_complete"] is False
    assert report["long_rare_positive_counts"] == {"therapy": 0, "ai": 0}
    assert {entry["kind"] for entry in report["shortfalls"]} == {"long_rare_positive"}


def test_subtopic_shortfalls_are_advisory_and_visible_with_complete_domain_coverage():
    cases = complete_fixture()
    for item in cases:
        if item["reference_domain"] == "ai":
            item["reference_subtopic"] = helpers.SUBTOPICS["ai"][0]
    report = coverage.coverage_report(cases)
    assert report["per_domain"]["ai"] == 25
    assert report["coverage_complete"] is True
    assert report["subtopic_balance_complete"] is False
    assert sum(entry["kind"] == "subtopic" for entry in report["advisory_shortfalls"]) == 4


def test_unresolved_or_wrong_pair_labels_never_receive_coverage_credit():
    cases = complete_fixture()
    cases[0]["reference_subtopic"] = None
    cases[1]["reference_subtopic"] = "relationships"
    cases[2]["reference_status"] = "unresolved_unscored"
    report = coverage.coverage_report(cases)
    assert report["positive_n"] == 98
    assert report["coverage_complete"] is False
    assert report["unassigned_reference_subtopic_counts"] == {"coding": 1}
    assert report["invalid_reference_ids"] == [cases[1]["conversation"]["hash"]]


def test_nonenglish_negative_cannot_be_mislabeled_as_a_positive_or_roleplay():
    cases = complete_fixture()
    changed = next(item for item in cases if item["negative_kind"] == "non_english")
    changed["reference_domain"] = "roleplay"
    report = coverage.coverage_report(cases)
    assert report["negative_kinds"]["non_english"] == 4
    assert changed["conversation"]["hash"] in report["invalid_reference_ids"]
    assert report["coverage_complete"] is False
