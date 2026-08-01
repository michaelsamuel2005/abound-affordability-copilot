"""Evaluation-harness tests: the metrics compute correctly on the held-out set,
the faithfulness checker actually catches fabrications, and the golden labels
stay consistent with the rules engine (regression guard for the whole system)."""

import copy

from agent import assess
from config import thresholds
from evaluate import _check_faithfulness, evaluate
from schemas import Outcome


def test_full_holdout_evaluation_regression(eval_data, eval_truth, retriever):
    """The headline numbers for deterministic mode. If a threshold, prompt,
    policy or generator change breaks any of these, CI fails."""
    apps, tx_by = eval_data
    ev = evaluate(apps, tx_by, retriever, truth=eval_truth)
    s = ev["summary"]
    assert s["dataset"]["n_applicants"] == 32
    assert s["decision"]["accuracy"] == 1.0
    assert s["guardrails"]["recall"] == 1.0
    assert s["retrieval"]["hit_rate_at_k"] == 1.0
    assert s["faithfulness"]["rate"] == 1.0
    assert s["categorization"]["accuracy"] >= 0.90
    assert s["categorization"]["critical_income_inflation_errors"] == 0
    assert s["structured_output"]["mode"] == "rules"
    assert s["latency_ms"]["end_to_end"]["p95"] < 1000


def test_confusion_and_per_class_shapes(eval_data, eval_truth, retriever):
    apps, tx_by = eval_data
    ev = evaluate(apps[:6], tx_by, retriever, truth=eval_truth)
    s = ev["summary"]
    assert sum(s["decision"]["confusion"].values()) == 6
    assert set(s["decision"]["per_class"]) == {"approve", "refer", "decline"}
    assert len(ev["rows"]) == 6 and all("end_to_end_ms" in r for r in ev["rows"])


def _one(eval_data, retriever, profile="healthy_high"):
    apps, tx_by = eval_data
    a = next(x for x in apps if x["profile"] == profile)
    d, diag = assess(a, tx_by[a["applicant_id"]], retriever)
    return d, diag


def test_faithfulness_checker_passes_genuine_output(eval_data, retriever):
    d, diag = _one(eval_data, retriever)
    ok, problems = _check_faithfulness(d, diag, retriever, thresholds())
    assert ok, problems


def test_faithfulness_checker_catches_tampered_quote(eval_data, retriever):
    d, diag = _one(eval_data, retriever)
    bad = copy.deepcopy(d)
    bad.policy_citations[0].quote = "The policy definitely allows this loan."
    ok, problems = _check_faithfulness(bad, diag, retriever, thresholds())
    assert not ok and any("verbatim" in p for p in problems)


def test_faithfulness_checker_catches_invented_number(eval_data, retriever):
    d, diag = _one(eval_data, retriever)
    bad = copy.deepcopy(d)
    bad.rationale = "Approved because income is £99,999 per month [POL-002]."
    ok, problems = _check_faithfulness(bad, diag, retriever, thresholds())
    assert not ok and any("99999" in p.replace(",", "") for p in problems)


def test_faithfulness_checker_catches_uncited_rationale(eval_data, retriever):
    d, diag = _one(eval_data, retriever)
    bad = copy.deepcopy(d)
    bad.rationale = "This one is fine, trust me."
    ok, problems = _check_faithfulness(bad, diag, retriever, thresholds())
    assert not ok


def test_eval_labels_consistent_with_rules_on_ground_truth(eval_data, retriever):
    """Documented circularity check: scenario-intent labels must agree with the
    rules engine, so any drift between generator and policy is caught here."""
    apps, tx_by = eval_data
    for a in apps:
        d, _ = assess(a, tx_by[a["applicant_id"]], retriever)
        assert d.outcome == Outcome(a["expected_outcome"]), a["profile"]
