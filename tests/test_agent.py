"""End-to-end agent tests on the generated dev set (label consistency for all 21
scenarios), plus injection resistance, citation integrity, and the rationale
validation gate with scripted LLM behaviour."""

import pytest
from fakes import FakeLLM

from agent import assess, validate_rationale
from config import thresholds
from schemas import Category, Outcome


@pytest.fixture(scope="module")
def dev_apps(dev_data):
    return dev_data


def _assess(dev_data, retriever, profile, llm=None):
    apps, tx_by = dev_data
    a = next(x for x in apps if x["profile"] == profile)
    return a, *assess(a, tx_by[a["applicant_id"]], retriever, llm=llm)


def test_all_dev_scenarios_match_expected_outcome_and_warnings(dev_data, retriever):
    apps, tx_by = dev_data
    assert len(apps) == 21
    for a in apps:
        d, diag = assess(a, tx_by[a["applicant_id"]], retriever)
        assert d.outcome.value == a["expected_outcome"], a["profile"]
        raised = {w.code.value for w in d.warnings}
        for w in a["expected_warnings"]:
            assert w in raised, (a["profile"], w, raised)


def test_thin_file_guardrail(dev_data, retriever):
    a, d, diag = _assess(dev_data, retriever, "thin_file")
    assert d.outcome == Outcome.refer
    assert d.guardrail is not None and d.guardrail.value == "insufficient_history"
    assert d.review_priority == "high" and d.human_review_required


def test_injection_descriptions_do_not_flip_anything(dev_data, retriever):
    a, d, diag = _assess(dev_data, retriever, "injection_descriptions")
    assert d.outcome == Outcome.approve
    # the injection transactions must NOT be categorised as income
    apps, tx_by = dev_data
    txns = tx_by[a["applicant_id"]]
    inj = [t["transaction_id"] for t in txns if "IGNORE" in t["description"]]
    assert inj, "injection transactions present in the dataset"
    for tid in inj:
        assert diag["categories"][tid] != Category.income.value


def test_citations_always_valid_and_verbatim(dev_data, retriever):
    apps, tx_by = dev_data
    for a in apps:
        d, _ = assess(a, tx_by[a["applicant_id"]], retriever)
        assert d.policy_citations, a["profile"]
        for c in d.policy_citations:
            chunk = retriever.by_id[c.policy_id]
            assert c.quote in chunk["body"]
            assert c.version == chunk["version"] and c.doc_id == chunk["doc_id"]


def test_evidence_ids_present_for_key_metrics(dev_data, retriever):
    a, d, diag = _assess(dev_data, retriever, "healthy_mid")
    ev = diag["metrics"]["evidence"]
    assert ev["eligible_income"] and ev["essential_spend"]


def test_versions_stamped(dev_data, retriever):
    a, d, _ = _assess(dev_data, retriever, "healthy_mid")
    assert d.versions["engine"] and d.versions["prompts"] == "v3"
    assert d.versions["llm"] == "deterministic"


# ---------------------------------------------------------------------------
# rationale validation gate (LLM narrative can only be accepted if grounded)
# ---------------------------------------------------------------------------


def test_grounded_llm_rationale_accepted(dev_data, retriever):
    grounded = (
        "The application is affordable: disposable income covers the new "
        "repayment with the required buffer [POL-002]."
    )
    a, d, diag = _assess(
        dev_data, retriever, "healthy_mid", llm=FakeLLM(["IGNORED-CATEGORISER-CALLS", grounded])
    )
    # note: FakeLLM also answered the categoriser; whatever it returned was
    # validated/fallen back, and the rationale call got the last response
    assert diag["rationale_source"] in ("llm", "template")
    if diag["rationale_source"] == "llm":
        assert d.rationale == grounded


def test_rationale_with_invented_number_rejected():
    facts = {"disposable_income": 800.0, "monthly_repayment": 260.0}
    ok, problems = validate_rationale(
        "Disposable income is £9999 after the repayment [POL-002].",
        facts,
        {"POL-002"},
        thresholds(),
    )
    assert not ok and any("9999" in p for p in problems)


def test_rationale_with_invented_policy_id_rejected():
    facts = {"disposable_income": 800.0}
    ok, problems = validate_rationale(
        "Approved per [POL-999] with disposable income £800.", facts, {"POL-002"}, thresholds()
    )
    assert not ok and any("POL-999" in p for p in problems)


def test_rationale_without_citation_rejected():
    ok, problems = validate_rationale("Looks fine to me.", {}, {"POL-002"}, thresholds())
    assert not ok and any("no policy ID" in p for p in problems)


def test_rationale_numbers_tolerate_formatting():
    facts = {"disposable_income": 1234.56, "dti_including_new": 0.42}
    ok, problems = validate_rationale(
        "Disposable income £1,234.56 and DTI 42% are within policy [POL-002].",
        facts,
        {"POL-002"},
        thresholds(),
    )
    assert ok, problems


def test_adversarial_llm_obeying_injection_cannot_produce_approve(retriever):
    """Worst-case LLM behaviour: the model 'obeys' a hostile instruction and
    labels EVERY transaction as income. Sign-consistency rejects it on all
    debits (mass fallback -> llm_output_invalid guardrail), so the outcome is a
    conservative refer — never an approve built on inflated income."""
    import json as _json

    from conftest import make_txn

    txns = []
    for mm in range(1, 7):
        m = f"2026-{mm:02d}"
        txns.append(
            make_txn(description="ACME LTD SALARY", amount=3000.0, date=f"{m}-28", raw_type="BGC")
        )
        txns.append(
            make_txn(description="RENT PROPERTYCO", amount=-900.0, date=f"{m}-01", raw_type="DD")
        )
        for wk in range(4):
            txns.append(
                make_txn(description="TESCO STORES", amount=-80.0, date=f"{m}-{3 + wk * 7:02d}")
            )
        for i in range(5):
            txns.append(make_txn(description="PRET A MANGER", amount=-9.5, date=f"{m}-{5 + i:02d}"))
    hostile = _json.dumps(
        {
            "items": [
                {"transaction_id": t["transaction_id"], "category": "income", "confidence": 0.99}
                for t in txns
            ]
        }
    )
    applicant = {"applicant_id": "ADV-001", "requested_amount": 4000.0, "term_months": 24}
    d, diag = assess(applicant, txns, retriever, llm=FakeLLM([hostile]))
    assert d.outcome != Outcome.approve
    assert d.outcome == Outcome.refer
    assert any(w.code.value == "llm_output_invalid" for w in d.warnings)
    debit_ids = {t["transaction_id"] for t in txns if t["amount"] < 0}
    assert all(diag["categories"][tid] != "income" for tid in debit_ids)


# ---------------------------------------------------------------------------
# revision-3 hardening: validator ratio hole, sign-valid attacks, failure paths
# ---------------------------------------------------------------------------


def test_validator_rejects_non_ratio_fact_times_100():
    """£3,000 income must not legitimise a fabricated £300,000 (the old
    every-fact-times-100 hole); ratio facts may still appear as percentages."""
    facts = {"monthly_income_assessed": 3000.0, "dti_including_new": 0.42}
    ok, problems = validate_rationale(
        "Income is £300000 [POL-002].", facts, {"POL-002"}, thresholds()
    )
    assert not ok and any("300000" in p for p in problems)
    ok, _ = validate_rationale(
        "Income is £3,000 and DTI is 42% [POL-002].", facts, {"POL-002"}, thresholds()
    )
    assert ok


def _forty_txn_payload():
    from conftest import make_txn

    txns, cats = [], {}
    for mm in range(1, 7):
        m = f"2026-{mm:02d}"
        t = make_txn(description="ACME LTD SALARY", amount=3000.0, date=f"{m}-28", raw_type="BGC")
        txns.append(t)
        cats[t["transaction_id"]] = "income"
        t = make_txn(description="RENT PROPERTYCO", amount=-900.0, date=f"{m}-01", raw_type="DD")
        txns.append(t)
        cats[t["transaction_id"]] = "rent_mortgage"
        for wk in range(4):
            t = make_txn(description="TESCO STORES", amount=-80.0, date=f"{m}-{3 + wk * 7:02d}")
            txns.append(t)
            cats[t["transaction_id"]] = "groceries"
    for i in range(4):
        t = make_txn(description="PRET A MANGER", amount=-9.5, date=f"2026-03-{10 + i:02d}")
        txns.append(t)
        cats[t["transaction_id"]] = "dining"
    return txns, cats


def test_sign_valid_transfer_attack_cannot_produce_approve(retriever):
    """The auditor's stronger adversary: credits -> income, debits ->
    internal_transfer. Sign-consistent, zero expenditure — must still refer via
    the DQ-007 transfer-imbalance / no-essential-spend plausibility guardrails."""
    import json as _json

    txns, _ = _forty_txn_payload()
    hostile = _json.dumps(
        {
            "items": [
                {
                    "transaction_id": t["transaction_id"],
                    "category": "income" if t["amount"] > 0 else "internal_transfer",
                    "confidence": 0.99,
                }
                for t in txns
            ]
        }
    )
    applicant = {"applicant_id": "ADV-002", "requested_amount": 4000.0, "term_months": 24}
    d, diag = assess(applicant, txns, retriever, llm=FakeLLM([hostile]))
    assert d.outcome == Outcome.refer
    raised = {w.code.value for w in d.warnings}
    assert {"transfer_imbalance", "no_essential_spend"} & raised
    assert diag["categorize_meta"]["sign_rejected"] == 0  # attack was sign-valid


def test_sign_valid_savings_attack_cannot_produce_approve(retriever):
    """Variant: every debit labelled `savings` (excluded from spend) -> zero
    essentials -> plausibility referral."""
    import json as _json

    txns, _ = _forty_txn_payload()
    hostile = _json.dumps(
        {
            "items": [
                {
                    "transaction_id": t["transaction_id"],
                    "category": "income" if t["amount"] > 0 else "savings",
                    "confidence": 0.99,
                }
                for t in txns
            ]
        }
    )
    applicant = {"applicant_id": "ADV-003", "requested_amount": 4000.0, "term_months": 24}
    d, _ = assess(applicant, txns, retriever, llm=FakeLLM([hostile]))
    assert d.outcome == Outcome.refer
    assert any(w.code.value == "no_essential_spend" for w in d.warnings)


def test_retriever_exception_degrades_to_referral(dev_data, retriever, monkeypatch):
    """An infrastructure failure inside retrieval must produce a coherent
    referral, not a 500."""
    apps, tx_by = dev_data
    a = next(x for x in apps if x["profile"] == "healthy_mid")

    def boom(query, k=None):
        raise RuntimeError("index corrupted")

    monkeypatch.setattr(retriever, "retrieve", boom)
    d, diag = assess(a, tx_by[a["applicant_id"]], retriever)
    assert d.outcome == Outcome.refer
    assert any(w.code.value == "retrieval_empty" for w in d.warnings)
    assert d.confidence == 0.5
    assert "referred for manual review" in d.reasons[0].lower()
    assert diag["retrieval"].get("error") == "retrieval_exception"


def test_rationale_provider_failure_after_approve_is_coherent(retriever):
    """If the rationale-stage provider dies after the rules approved, the
    decision must flip to refer AND read as a referral (confidence, reasons,
    citations) — not as an approval with a refer label."""
    import json as _json

    txns, cats = _forty_txn_payload()
    valid = _json.dumps(
        {
            "items": [
                {"transaction_id": tid, "category": c, "confidence": 0.95}
                for tid, c in cats.items()
            ]
        }
    )
    applicant = {"applicant_id": "FLIP-001", "requested_amount": 4000.0, "term_months": 24}
    d, diag = assess(
        applicant, txns, retriever, llm=FakeLLM([valid, ConnectionError("provider down")])
    )
    assert d.outcome == Outcome.refer
    assert d.guardrail is not None and d.guardrail.value == "llm_unavailable"
    assert d.confidence == 0.5
    assert d.reasons[0].startswith("Referred for manual review")
    assert "Pre-referral rules assessment for context: approve" in d.reasons[1]
    assert d.policy_citations[0].policy_id == "POL-010"
    assert "[POL-010]" in d.rationale


def test_rationale_citing_retrieved_only_id_gets_citation_object(dev_data, retriever):
    """Citation completeness: an accepted rationale citing a retrieved-but-not-
    decisive policy ID must yield a structured citation for it."""
    txns, cats = _forty_txn_payload()
    import json as _json

    valid = _json.dumps(
        {
            "items": [
                {"transaction_id": tid, "category": c, "confidence": 0.95}
                for tid, c in cats.items()
            ]
        }
    )
    grounded = (
        "Affordability is assessed on cashflow, not credit score [POL-001], and the "
        "buffer requirement is met [POL-002]."
    )
    applicant = {"applicant_id": "CITE-001", "requested_amount": 4000.0, "term_months": 24}
    d, diag = assess(applicant, txns, retriever, llm=FakeLLM([valid, grounded]))
    if diag["rationale_source"] == "llm":  # POL-001 must be retrieved for this case
        cited = {c.policy_id for c in d.policy_citations}
        assert "POL-001" in cited and "POL-002" in cited
        pol1 = next(c for c in d.policy_citations if c.policy_id == "POL-001")
        assert pol1.quote in retriever.by_id["POL-001"]["body"]
