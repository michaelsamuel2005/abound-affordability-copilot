"""Deterministic decision rules: the ONLY component that chooses the outcome.

The LLM never selects approve/refer/decline — it categorises transactions and
writes the rationale. This module applies the policy thresholds from `config.py`
in a FIXED, documented precedence and returns machine-readable warnings.

Precedence (first match wins for the outcome; all warnings are still collected):
  1. Pipeline-integrity + data-sufficiency guardrails      -> refer (never decline)
  2. DTI including the new loan > 45%                      -> decline   [POL-003]
  3. Gambling > 10% of assessed income                     -> refer     [POL-005]
  4. >= 2 distress events                                  -> refer     [POL-006]
  5. Income volatility (cv) > 0.35                         -> refer     [POL-004]
  6. Post-repayment buffer < £150:
       reduced amount viable (>= £500 and >= 50% of ask)   -> refer     [POL-002, POL-009]
       otherwise                                           -> decline   [POL-002, POL-009]
  7. DTI in the 40–45% band                                -> refer     [POL-003]
  8. Vulnerability indicators (benefits > 50% of income
     AND essentials > 60% of income)                       -> refer     [POL-008]
  9. Otherwise                                             -> approve   [POL-002]

Design invariants (unit-tested):
  * a guardrail can never produce an automatic decline — insufficient or
    unreliable data always routes to a human;
  * technical failures (LLM down / invalid output / empty retrieval) also route
    to refer, never decline;
  * every outcome carries at least one policy citation ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import Thresholds
from schemas import AffordabilityMetrics, Outcome, WarningCode, WarningFlag, make_warning

# policy sections that justify each guardrail
GUARDRAIL_POLICY = {
    WarningCode.insufficient_history: "POL-007",
    WarningCode.low_transaction_count: "POL-007",
    WarningCode.no_recognisable_income: "DQ-006",
    WarningCode.coverage_gap: "DQ-006",
    WarningCode.high_unknown_share: "DQ-001",
    WarningCode.high_cash_usage: "DQ-005",
    WarningCode.retrieval_empty: "POL-010",
    WarningCode.llm_output_invalid: "POL-010",
    WarningCode.llm_unavailable: "POL-010",
    WarningCode.transfer_imbalance: "DQ-007",
    WarningCode.no_essential_spend: "DQ-007",
}


@dataclass
class RuleResult:
    outcome: Outcome
    confidence: float
    guardrail: WarningCode | None
    warnings: list[WarningFlag] = field(default_factory=list)
    decisive_policy_ids: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _data_guardrails(m: AffordabilityMetrics, cfg: Thresholds) -> list[WarningFlag]:
    w: list[WarningFlag] = []
    if m.months_observed < cfg.min_months:
        w.append(
            make_warning(
                WarningCode.insufficient_history,
                f"Only {m.months_observed} month(s) of history; minimum is {cfg.min_months}.",
            )
        )
    if m.n_transactions < cfg.min_transactions:
        w.append(
            make_warning(
                WarningCode.low_transaction_count,
                f"Only {m.n_transactions} transactions; minimum is {cfg.min_transactions}.",
            )
        )
    if m.monthly_income_assessed <= 0 or m.income_months < cfg.min_income_months:
        w.append(
            make_warning(
                WarningCode.no_recognisable_income,
                f"Eligible income recognised in {m.income_months} month(s); "
                f"minimum is {cfg.min_income_months}.",
            )
        )
    if m.coverage_gap:
        w.append(
            make_warning(
                WarningCode.coverage_gap, "Observation window is non-contiguous (missing month)."
            )
        )
    if m.unknown_share > cfg.max_unknown_share:
        w.append(
            make_warning(
                WarningCode.high_unknown_share,
                f"Unclassified transactions are {m.unknown_share:.0%} of debit "
                f"value (limit {cfg.max_unknown_share:.0%}).",
            )
        )
    if m.cash_share > cfg.max_cash_share:
        w.append(
            make_warning(
                WarningCode.high_cash_usage,
                f"Cash withdrawals are {m.cash_share:.0%} of debit value "
                f"(limit {cfg.max_cash_share:.0%}).",
            )
        )
    # DQ-007 classification-plausibility checks: a hostile or broken categoriser
    # can be sign-consistent yet implausible (e.g. every debit labelled as an
    # own-account transfer). These deterministic checks force a referral.
    #
    # The leg-balance test is only ANSWERABLE when both legs could be observed,
    # i.e. when at least two accounts are connected. On a single-account applicant
    # a transfer to an external savings account has no visible counterpart, so an
    # imbalance is the expected reading, not a suspicious one. Before this gate the
    # check fired on every honest single-account saver: in the live-LLM run of
    # 2026-08-01 it referred 7 of 11 approve-labelled applicants (false-positive
    # referral rate 0.636) because the model reasonably labelled "TRANSFER TO
    # SAVINGS" as internal_transfer. Real Open Banking coverage is frequently
    # partial, so the assumption was wrong in production terms too.
    #
    # Residual risk, stated plainly: on single-account applicants a sign-valid
    # hostile categoriser can still hide debits behind `internal_transfer` without
    # tripping this check. The remaining defences for that population are the
    # no_essential_spend check below, the measured critical-error counts, and human
    # review — not this rule.
    if m.n_accounts >= 2 and m.n_internal_transfer_txns and m.internal_transfer_gross > 0:
        tol = max(
            cfg.transfer_imbalance_floor_gbp,
            cfg.transfer_imbalance_tolerance * m.internal_transfer_gross,
        )
        if abs(m.internal_transfer_net) > tol:
            w.append(
                make_warning(
                    WarningCode.transfer_imbalance,
                    f"Internal-transfer legs do not balance (net £"
                    f"{m.internal_transfer_net:,.0f} vs gross £"
                    f"{m.internal_transfer_gross:,.0f}); classification unreliable.",
                )
            )
    if m.essential_spend == 0 and m.n_transactions >= cfg.min_transactions:
        w.append(
            make_warning(
                WarningCode.no_essential_spend,
                f"No essential spending recognised across {m.n_transactions} "
                "transactions; classification implausible.",
            )
        )
    return w


def _pipeline_guardrails(
    categorize_meta: dict, n_txns: int, retrieval_empty: bool
) -> list[WarningFlag]:
    w: list[WarningFlag] = []
    if categorize_meta.get("llm_unavailable"):
        w.append(
            make_warning(
                WarningCode.llm_unavailable,
                "LLM provider unavailable; categorisation fell back to rules.",
            )
        )
    if n_txns and categorize_meta.get("fallback_items", 0) > 0.2 * n_txns:
        w.append(
            make_warning(
                WarningCode.llm_output_invalid,
                f"{categorize_meta['fallback_items']} of {n_txns} categorisations "
                "failed structured-output validation and used the rule fallback.",
            )
        )
    if retrieval_empty:
        w.append(
            make_warning(
                WarningCode.retrieval_empty,
                "No policy passage retrieved above the similarity threshold.",
            )
        )
    if categorize_meta.get("repairs", 0) > 0:
        w.append(
            make_warning(
                WarningCode.llm_output_repaired,
                f"{categorize_meta['repairs']} categorisation batch(es) needed a "
                "repair prompt before validating.",
            )
        )
    return w


def _info_treatments(m: AffordabilityMetrics) -> tuple[list[WarningFlag], list[str]]:
    w, pids = [], []
    if m.n_duplicates_removed:
        w.append(
            make_warning(
                WarningCode.duplicates_removed,
                f"{m.n_duplicates_removed} duplicate posting(s) removed before calculation.",
            )
        )
        pids.append("DQ-004")
    if m.n_internal_transfer_txns:
        w.append(
            make_warning(
                WarningCode.internal_transfers_netted,
                f"{m.n_internal_transfer_txns} own-account transfer leg(s) excluded "
                "from income and expenditure.",
            )
        )
        pids.append("DQ-002")
    if m.n_refunds_netted:
        w.append(
            make_warning(
                WarningCode.refunds_netted,
                f"{m.n_refunds_netted} refund(s) netted against original spend.",
            )
        )
        pids.append("DQ-003")
    return w, pids


def evaluate_rules(
    m: AffordabilityMetrics,
    requested: float,
    term: int,
    repayment: float,
    disposable_after: float,
    dti_new: float,
    max_affordable: float,
    cfg: Thresholds,
    categorize_meta: dict | None = None,
    retrieval_empty: bool = False,
) -> RuleResult:
    categorize_meta = categorize_meta or {}
    warnings = _pipeline_guardrails(categorize_meta, m.n_transactions, retrieval_empty)
    warnings += _data_guardrails(m, cfg)
    info_w, info_pids = _info_treatments(m)

    guard = next((w for w in warnings if w.severity == "guardrail"), None)
    pids: list[str] = []
    reasons: list[str] = []

    if guard is not None:
        outcome, conf = Outcome.refer, 0.5
        for w in warnings:
            if w.severity == "guardrail":
                pid = GUARDRAIL_POLICY[w.code]
                if pid not in pids:
                    pids.append(pid)
        reasons.append(
            "Automated affordability assessment is not reliable for this "
            "application; referred for manual review: "
            + "; ".join(w.message for w in warnings if w.severity == "guardrail")
        )
    elif dti_new > cfg.dti_max:
        outcome, conf = Outcome.decline, 0.9
        pids.append("POL-003")
        warnings.append(
            make_warning(
                WarningCode.dti_borderline, f"DTI including the new loan is {dti_new:.0%}."
            )
        )
        reasons.append(
            f"Debt-to-income including the new loan is {dti_new:.0%}, above the "
            f"{cfg.dti_max:.0%} limit."
        )
    elif m.gambling_ratio > cfg.gambling_refer:
        outcome, conf = Outcome.refer, 0.65
        pids.append("POL-005")
        warnings.append(
            make_warning(
                WarningCode.gambling_high, f"Gambling is {m.gambling_ratio:.0%} of assessed income."
            )
        )
        reasons.append(
            f"Gambling spend is {m.gambling_ratio:.0%} of assessed income "
            f"(limit {cfg.gambling_refer:.0%}); manual review required."
        )
    elif m.distress_events >= cfg.distress_max:
        outcome, conf = Outcome.refer, 0.65
        pids.append("POL-006")
        warnings.append(
            make_warning(
                WarningCode.distress_events,
                f"{m.distress_events} financial-distress events observed.",
            )
        )
        reasons.append(
            f"{m.distress_events} distress events (overdraft/returned-DD fees) "
            "observed; manual review required."
        )
    elif m.income_volatility > cfg.volatility_max:
        outcome, conf = Outcome.refer, 0.65
        pids.append("POL-004")
        warnings.append(
            make_warning(
                WarningCode.income_volatility_high,
                f"Income volatility cv={m.income_volatility:.2f}.",
            )
        )
        reasons.append(
            f"Income is volatile (cv {m.income_volatility:.2f} > "
            f"{cfg.volatility_max:.2f}); assessed conservatively at "
            f"£{m.monthly_income_assessed:,.0f}/month — refer for verification."
        )
    elif disposable_after < cfg.buffer_gbp:
        pids += ["POL-002", "POL-009"]
        viable = (
            max_affordable >= cfg.reduced_offer_min_gbp
            and max_affordable >= cfg.reduced_offer_min_fraction * requested
        )
        if viable:
            outcome, conf = Outcome.refer, 0.65
            warnings.append(
                make_warning(
                    WarningCode.buffer_breach_reduced_offer,
                    f"Buffer after repayment £{disposable_after:,.0f} "
                    f"< £{cfg.buffer_gbp:,.0f}; reduced amount viable.",
                )
            )
            reasons.append(
                f"At £{requested:,.0f} the post-repayment buffer falls to "
                f"£{disposable_after:,.0f} (< £{cfg.buffer_gbp:,.0f}); up to about "
                f"£{max_affordable:,.0f} is affordable — refer to offer a reduced "
                "amount."
            )
        else:
            outcome, conf = Outcome.decline, 0.9
            reasons.append(
                f"At £{requested:,.0f} the post-repayment buffer falls to "
                f"£{disposable_after:,.0f} (< £{cfg.buffer_gbp:,.0f}) and the maximum "
                f"affordable (~£{max_affordable:,.0f}) is far below the request."
            )
    elif cfg.dti_refer <= dti_new <= cfg.dti_max:
        outcome, conf = Outcome.refer, 0.65
        pids.append("POL-003")
        warnings.append(
            make_warning(
                WarningCode.dti_borderline,
                f"DTI including the new loan is {dti_new:.0%} "
                f"({cfg.dti_refer:.0%}–{cfg.dti_max:.0%} band).",
            )
        )
        reasons.append(
            f"Debt-to-income including the new loan is {dti_new:.0%}, inside the "
            f"{cfg.dti_refer:.0%}–{cfg.dti_max:.0%} manual-review band."
        )
    elif (
        m.benefits_share > cfg.benefits_share_review
        and m.essential_share > cfg.essential_share_high
    ):
        outcome, conf = Outcome.refer, 0.65
        pids.append("POL-008")
        warnings.append(
            make_warning(
                WarningCode.vulnerability_indicators,
                f"Benefits are {m.benefits_share:.0%} of income and "
                f"essentials {m.essential_share:.0%} of income.",
            )
        )
        reasons.append(
            f"Potential vulnerability indicators: benefits are "
            f"{m.benefits_share:.0%} of income and essential spend is "
            f"{m.essential_share:.0%} of income — manual review required."
        )
    else:
        outcome, conf = Outcome.approve, 0.9
        pids.append("POL-002")
        reasons.append(
            f"Affordable: disposable income £{m.disposable_income:,.0f}/month, "
            f"buffer after repayment £{disposable_after:,.0f} "
            f"(≥ £{cfg.buffer_gbp:,.0f}), DTI including the new loan {dti_new:.0%}."
        )

    warnings += info_w
    for pid in info_pids:
        if pid not in pids:
            pids.append(pid)

    return RuleResult(
        outcome=outcome,
        confidence=conf,
        guardrail=guard.code if guard else None,
        warnings=warnings,
        decisive_policy_ids=pids,
        reasons=reasons,
    )
