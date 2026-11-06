"""Prospective reference coverage targets, separate from classifier agreement."""

from collections import Counter

from wild.selection import helpers as selection

NEGATIVE_KINDS = ("mixed", "roleplay", "non_english", "other")
RARE_DOMAINS = ("therapy", "ai")


def validation_plan():
    """Explicit coverage goals, not a statistical-power or accuracy guarantee."""
    return {
        "schema_version": 1,
        "policy_version": selection.POLICY_VERSION,
        "policy_sha256": selection.checksum(selection.POLICY.encode()),
        "taxonomy": selection.taxonomy_metadata(),
        "eligible_split": "heldout",
        "positive_target": 100,
        "per_domain_target": 25,
        "per_subtopic_target": 5,
        "subtopic_target_is_advisory": True,
        "negative_target": 20,
        "negative_kind_targets": dict.fromkeys(NEGATIVE_KINDS, 5),
        "long_window": {"min_turns": 13, "max_turns": 15, "per_rare_domain_target": 5},
        "retained_window": {"min_turns": 3, "max_turns": 15},
        "interpretation": "Deliberate domain and long-window coverage targets, not a power calculation or a population error-rate estimate. Five examples per subtopic is an advisory balance goal; genuine subtopic scarcity does not block domain validation. Report all shortfalls without manufacturing turns or broadening focused domains.",
        "reference_review": "Read every original retained user turn. AI-assisted reference labels remain provisional unless the user actually adjudicates them; model results measure agreement, not human accuracy.",
        "source_separation": "Exclude the prior 140-case audit, all 47 v2 references, all previously screened cases, and the old 8284-case pool by identity, transcript prefix, and source-turn identity. Prefer new pinned source shards. Never promote prompt-development examples into heldout.",
        "freeze": "Freeze policy, taxonomy, references, provenance, exclusions, and review statuses before new Llama predictions; no projection or harmfulness outcomes may inform sampling or labels.",
        "sampling": "Prospective keyword enrichment is permitted only as observable source retrieval, with recorded search strata and all-turn review. Same-domain topic changes remain eligible under the dominant-subtopic rule.",
    }


def coverage_report(cases):
    """Count only reviewed prospective heldout references toward fixed goals."""
    plan = validation_plan()
    heldout = [case for case in cases if case.get("split") == "heldout"]
    reviewed = [case for case in heldout if case.get("reference_status") in {"ai_reviewed_provisional", "human_adjudicated"}]
    topics = {domain: Counter({topic: 0 for topic in selection.SUBTOPICS[domain]}) for domain in selection.DOMAINS}
    negatives, long_windows, turns, statuses, domains = Counter(), Counter(), Counter(), Counter(), Counter()
    missing_subtopics = Counter()
    invalid_references = []
    seen = set()
    for case in reviewed:
        key = case["conversation"]["hash"]
        domain, topic = case.get("reference_domain"), case.get("reference_subtopic")
        count = len(case["conversation"]["users"])
        if key in seen or not 3 <= count <= 15:
            invalid_references.append(key)
            continue
        seen.add(key)
        turns[count] += 1
        statuses[case["reference_status"]] += 1
        if domain in selection.DOMAINS:
            if topic is None:
                missing_subtopics[domain] += 1
            elif topic not in selection.SUBTOPICS[domain]:
                invalid_references.append(case["conversation"]["hash"])
            else:
                topics[domain][topic] += 1
            if topic is None or topic in selection.SUBTOPICS[domain]:
                domains[domain] += 1
                if domain in RARE_DOMAINS and 13 <= count <= 15:
                    long_windows[domain] += 1
        elif domain in {"mixed", "roleplay", "other"} and topic is None:
            kind = case.get("negative_kind")
            if kind in NEGATIVE_KINDS and domain == ("other" if kind == "non_english" else kind):
                negatives[kind] += 1
            else:
                invalid_references.append(case["conversation"]["hash"])
        else:
            invalid_references.append(case["conversation"]["hash"])
    shortfalls, advisory_shortfalls = [], []
    for domain in selection.DOMAINS:
        if domains[domain] < plan["per_domain_target"]:
            shortfalls.append({"kind": "domain", "domain": domain, "observed": domains[domain], "target": 25, "missing": 25 - domains[domain]})
        for topic in selection.SUBTOPICS[domain]:
            observed = topics[domain][topic]
            if observed < plan["per_subtopic_target"]:
                advisory_shortfalls.append({"kind": "subtopic", "domain": domain, "subtopic": topic, "observed": observed, "target": 5, "missing": 5 - observed})
    for kind in NEGATIVE_KINDS:
        if negatives[kind] < 5:
            shortfalls.append({"kind": "negative", "negative_kind": kind, "observed": negatives[kind], "target": 5, "missing": 5 - negatives[kind]})
    for domain in RARE_DOMAINS:
        if long_windows[domain] < 5:
            shortfalls.append({"kind": "long_rare_positive", "domain": domain, "observed": long_windows[domain], "target": 5, "missing": 5 - long_windows[domain]})
    return {
        "coverage_complete": not shortfalls and not invalid_references,
        "targets": plan,
        "reference_n": len(cases), "heldout_n": len(heldout), "reviewed_heldout_n": len(reviewed),
        "excluded_from_coverage_n": len(cases) - len(reviewed),
        "positive_n": sum(domains.values()),
        "per_domain": {domain: domains[domain] for domain in selection.DOMAINS},
        "per_subtopic": {domain: dict(counts) for domain, counts in topics.items()},
        "negative_n": sum(negatives.values()),
        "negative_kinds": {kind: negatives[kind] for kind in NEGATIVE_KINDS},
        "long_rare_positive_counts": {domain: long_windows[domain] for domain in RARE_DOMAINS},
        "unassigned_reference_subtopic_counts": dict(missing_subtopics),
        "invalid_reference_ids": sorted(invalid_references),
        "retained_turn_counts": dict(sorted(turns.items())),
        "reference_status_counts": dict(statuses),
        "shortfalls": shortfalls,
        "advisory_shortfalls": advisory_shortfalls,
        "subtopic_balance_complete": not advisory_shortfalls,
    }


def readiness_report(cases, *, references_frozen=False, source_isolation_verified=False,
                     classifier_execution_complete=False, disagreement_review_complete=False,
                     code_checks_complete=False):
    report = coverage_report(cases)
    validation_ready = code_checks_complete and report["coverage_complete"] and references_frozen and source_isolation_verified
    return {
        "code_checks_complete": code_checks_complete,
        "reference_coverage_complete": report["coverage_complete"],
        "references_frozen": references_frozen,
        "source_isolation_verified": source_isolation_verified,
        "ready_for_classifier_diagnostics": references_frozen and source_isolation_verified,
        "ready_for_classifier_validation": validation_ready,
        "classifier_execution_complete": classifier_execution_complete,
        "disagreement_review_complete": disagreement_review_complete,
        "classifier_validation_complete": validation_ready and classifier_execution_complete and disagreement_review_complete,
        "study_quality_approval": "not_assessed",
        "interpretation": "Executable classifier diagnostics, code checks, reference coverage, and reviewed classifier validation are separate milestones. An incomplete reference set can produce diagnostic agreement metrics. No accuracy threshold or automatic study approval is inferred from provisional-reference agreement.",
        "coverage": report,
    }
