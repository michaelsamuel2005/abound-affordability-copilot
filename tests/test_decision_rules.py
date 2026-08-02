"""Decision-rule tests at EXACT threshold boundaries, plus the safety invariants:
guardrails always refer (never decline), technical failures never decline, and
precedence is fixed."""

from config import thresholds
from decision_rules import evaluate_rules
from schemas import AffordabilityMetrics, Outcome, WarningCode

CFG = thresholds()


def make_metrics(**over) -> AffordabilityMetrics:
    base = dict(
        months_observed=6,
        coverage_gap=False,
        n_transactions=150,
        n_duplicates_removed=0,
        n_internal_transfer_txns=0,
        n_refunds_netted=0,
        monthly_income_mean=3000.0,
        monthly_income_assessed=3000.0,
        income_volatility=0.05,
        income_months=6,
        income_sources=1,
        benefits_share=0.0,
        essential_spend=1400.0,
        discretionary_spend=300.0,
        existing_debt_repayments=200.0,
        gambling_spend=0.0,
        cash_withdrawals=0.0,
        unknown_spend=0.0,
        savings_transfers=0.0,
        unknown_share=0.02,
        cash_share=0.0,
        essential_share=0.47,
        gambling_ratio=0.0,
        dti_existing=0.067,
        disposable_income=1400.0,
        distress_events=0,
    )
    base.update(over)
    return AffordabilityMetrics(**base)


def run(
    m,
    requested=5000.0,
    term=24,
    repayment=260.0,
    disposable_after=None,
    dti=0.15,
    max_afford=8000.0,
    cat_meta=None,
    retrieval_empty=False,
):
    disposable_after = (
        m.disposable_income - repayment if disposable_after is None else disposable_after
    )
    return evaluate_rules(
        m,
        requested,
        term,
        repayment,
        disposable_after,
        dti,
        max_afford,
        CFG,
        categorize_meta=cat_meta or {},
        retrieval_empty=retrieval_empty,
    )


def codes(rr):
    return {w.code for w in rr.warnings}


# ---------------------------------------------------------------------------
# outcome boundaries (thresholds: buffer £150, DTI 40/45%, gambling 10%, cv .35)
# ---------------------------------------------------------------------------


def test_clean_case_approves_and_cites_buffer_policy():
    rr = run(make_metrics())
    assert rr.outcome == Outcome.approve and rr.confidence == 0.9
    assert "POL-002" in rr.decisive_policy_ids and rr.guardrail is None


def test_dti_boundaries():
    assert run(make_metrics(), dti=0.399).outcome == Outcome.approve
    assert run(make_metrics(), dti=0.400).outcome == Outcome.refer  # band inclusive
    assert run(make_metrics(), dti=0.450).outcome == Outcome.refer
    rr = run(make_metrics(), dti=0.451)
    assert rr.outcome == Outcome.decline and rr.decisive_policy_ids == ["POL-003"]


def test_buffer_boundaries():
    assert run(make_metrics(), disposable_after=150.0).outcome == Outcome.approve
    rr = run(make_metrics(), disposable_after=149.99, max_afford=4000.0)
    assert rr.outcome == Outcome.refer  # reduced offer viable
    assert WarningCode.buffer_breach_reduced_offer in codes(rr)
    assert set(rr.decisive_policy_ids) == {"POL-002", "POL-009"}


def test_buffer_breach_without_viable_reduction_declines():
    rr = run(make_metrics(), requested=10000.0, disposable_after=-50.0, max_afford=400.0)
    assert rr.outcome == Outcome.decline
    assert set(rr.decisive_policy_ids) == {"POL-002", "POL-009"}


def test_reduced_offer_needs_half_of_request_AND_500():
    # £900 affordable of a £10k ask: > £500 but < 50% -> decline
    rr = run(make_metrics(), requested=10000.0, disposable_after=100.0, max_afford=900.0)
    assert rr.outcome == Outcome.decline


def test_gambling_boundary():
    assert run(make_metrics(gambling_ratio=0.10)).outcome == Outcome.approve
    rr = run(make_metrics(gambling_ratio=0.101))
    assert rr.outcome == Outcome.refer and rr.decisive_policy_ids == ["POL-005"]


def test_distress_boundary():
    assert run(make_metrics(distress_events=1)).outcome == Outcome.approve
    rr = run(make_metrics(distress_events=2))
    assert rr.outcome == Outcome.refer and rr.decisive_policy_ids == ["POL-006"]


def test_volatility_boundary():
    assert run(make_metrics(income_volatility=0.35)).outcome == Outcome.approve
    rr = run(make_metrics(income_volatility=0.351))
    assert rr.outcome == Outcome.refer and rr.decisive_policy_ids == ["POL-004"]


def test_vulnerability_needs_both_conditions():
    assert run(make_metrics(benefits_share=0.6)).outcome == Outcome.approve
    assert run(make_metrics(essential_share=0.7)).outcome == Outcome.approve
    rr = run(make_metrics(benefits_share=0.6, essential_share=0.7))
    assert rr.outcome == Outcome.refer and rr.decisive_policy_ids == ["POL-008"]


# ---------------------------------------------------------------------------
# guardrails: each fires, all force refer, none can decline
# ---------------------------------------------------------------------------


def test_each_data_guardrail_fires():
    for over, code in [
        (dict(months_observed=2), WarningCode.insufficient_history),
        (dict(n_transactions=39), WarningCode.low_transaction_count),
        (dict(monthly_income_assessed=0.0, income_months=0), WarningCode.no_recognisable_income),
        (dict(coverage_gap=True), WarningCode.coverage_gap),
        (dict(unknown_share=0.101), WarningCode.high_unknown_share),
        (dict(cash_share=0.251), WarningCode.high_cash_usage),
    ]:
        rr = run(make_metrics(**over))
        assert rr.outcome == Outcome.refer, over
        assert code in codes(rr) and rr.guardrail is not None


def test_guardrail_beats_decline_worthy_dti():
    """Safety invariant: unreliable data can never produce an automatic decline."""
    rr = run(make_metrics(months_observed=1, n_transactions=10), dti=0.60)
    assert rr.outcome == Outcome.refer
    assert rr.guardrail == WarningCode.insufficient_history
    assert rr.confidence == 0.5


def test_technical_failures_refer_never_decline():
    rr = run(make_metrics(), dti=0.60, cat_meta={"llm_unavailable": True})
    assert rr.outcome == Outcome.refer
    assert WarningCode.llm_unavailable in codes(rr)

    rr = run(make_metrics(), dti=0.60, retrieval_empty=True)
    assert rr.outcome == Outcome.refer
    assert WarningCode.retrieval_empty in codes(rr)

    rr = run(
        make_metrics(n_transactions=100), dti=0.60, cat_meta={"fallback_items": 30}
    )  # >20% invalid outputs
    assert rr.outcome == Outcome.refer
    assert WarningCode.llm_output_invalid in codes(rr)


def test_repairs_are_informational_not_referral():
    rr = run(make_metrics(), cat_meta={"repairs": 1})
    assert rr.outcome == Outcome.approve
    assert WarningCode.llm_output_repaired in codes(rr)


def test_info_treatments_recorded_with_policy_ids():
    rr = run(make_metrics(n_duplicates_removed=1, n_internal_transfer_txns=4, n_refunds_netted=2))
    assert rr.outcome == Outcome.approve
    assert {
        WarningCode.duplicates_removed,
        WarningCode.internal_transfers_netted,
        WarningCode.refunds_netted,
    } <= codes(rr)
    assert {"DQ-002", "DQ-003", "DQ-004"} <= set(rr.decisive_policy_ids)


def test_every_outcome_carries_a_policy_id_and_reason():
    for over in (dict(), dict(months_observed=1), dict(gambling_ratio=0.2)):
        rr = run(make_metrics(**over))
        assert rr.decisive_policy_ids and rr.reasons


def test_precedence_dti_decline_before_gambling_refer():
    rr = run(make_metrics(gambling_ratio=0.2), dti=0.50)
    assert rr.outcome == Outcome.decline
    assert rr.decisive_policy_ids[0] == "POL-003"


def test_transfer_imbalance_guardrail():
    """Sign-valid hostile pattern: debits mass-labelled as own-account transfers
    leave grossly one-sided transfer legs -> refer, never approve."""
    rr = run(
        make_metrics(
            n_accounts=2,
            n_internal_transfer_txns=120,
            internal_transfer_net=-13000.0,
            internal_transfer_gross=13000.0,
        )
    )
    assert rr.outcome == Outcome.refer
    assert WarningCode.transfer_imbalance in codes(rr)
    assert "DQ-007" in rr.decisive_policy_ids


def test_single_account_one_sided_transfers_do_not_trigger_imbalance():
    """Regression (live-LLM run 2026-08-01): the leg-balance test is unanswerable on
    a single connected account — a transfer to an external savings account has no
    visible counterpart — and firing it referred 7 of 11 approve-labelled applicants
    (false-positive referral rate 0.636). Now gated on n_accounts >= 2."""
    rr = run(
        make_metrics(
            n_accounts=1,
            n_internal_transfer_txns=6,
            internal_transfer_net=-1800.0,
            internal_transfer_gross=1800.0,
        )
    )
    assert rr.outcome == Outcome.approve
    assert WarningCode.transfer_imbalance not in codes(rr)


def test_single_account_hostile_pattern_still_referred_by_essential_spend():
    """The gate must not open an approval route: the sign-valid attack that hides
    every debit behind `internal_transfer` leaves zero essential spend, and the
    second DQ-007 check still catches it on a single-account applicant."""
    rr = run(
        make_metrics(
            n_accounts=1,
            n_internal_transfer_txns=120,
            internal_transfer_net=-13000.0,
            internal_transfer_gross=13000.0,
            essential_spend=0.0,
            essential_share=0.0,
        )
    )
    assert rr.outcome == Outcome.refer
    assert WarningCode.no_essential_spend in codes(rr)


def test_balanced_transfers_do_not_trigger_imbalance():
    rr = run(
        make_metrics(
            n_internal_transfer_txns=12,
            internal_transfer_net=0.0,
            internal_transfer_gross=3000.0,
        )
    )
    assert rr.outcome == Outcome.approve
    assert WarningCode.transfer_imbalance not in codes(rr)


def test_no_essential_spend_guardrail():
    rr = run(make_metrics(essential_spend=0.0, essential_share=0.0))
    assert rr.outcome == Outcome.refer
    assert WarningCode.no_essential_spend in codes(rr)
    assert "DQ-007" in rr.decisive_policy_ids
