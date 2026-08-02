"""Deterministic affordability engine — "cashflow underwriting" in miniature.

All money arithmetic is INTEGER PENCE (converted via Decimal, never float maths),
so totals are exact and reproducible; floats appear only at the API boundary.
Every aggregate carries the transaction IDs behind it (`evidence`), so a reviewer
can reproduce any figure from raw data.

Data-quality passes, in order (each logged and evidenced):
  1. duplicate removal      — identical (account, date, description, amount, type)
  2. internal transfers     — both legs excluded from income AND expenditure
  3. savings movements      — excluded from expenditure (money moved, not spent)
  4. refund netting         — refunds matched to an earlier same-merchant debit
                              are netted against that category; unmatched refunds
                              are excluded entirely (never income)
  5. unknown handling       — unknown debits reduce disposable income
                              (conservative) and feed the unknown-share guardrail

Income rules: eligible income = `income` + `benefits` credits only. An
unrecognised inflow is NOT income. Volatile income (cv > threshold) is assessed
at a conservative figure: min(worst month, mean x (1 - cv)).

The engine never sees ground-truth labels and contains no ML: given the same
categorised transactions it always produces the same metrics.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal

from config import Thresholds, thresholds
from schemas import AffordabilityMetrics, CategorizedTransaction, Category

BUCKET_ESSENTIAL = {
    Category.rent_mortgage,
    Category.utilities,
    Category.council_tax,
    Category.groceries,
    Category.transport,
    Category.insurance,
}
BUCKET_DEBT = {Category.loan_repayment, Category.bnpl}
BUCKET_DISCRETIONARY = {
    Category.subscriptions,
    Category.dining,
    Category.shopping,
    Category.entertainment,
}
ELIGIBLE_INCOME = {Category.income, Category.benefits}
EXCLUDED_MOVEMENT = {Category.internal_transfer, Category.savings}

REFUND_MATCH_DAYS = 45  # refund may net against a debit up to 45 days earlier


# ---------------------------------------------------------------------------
# Money helpers
# ---------------------------------------------------------------------------


def to_pence(amount: float | str | int) -> int:
    """Exact conversion GBP -> integer pence via Decimal (never float maths)."""
    return int((Decimal(str(amount)) * 100).to_integral_value(rounding=ROUND_HALF_UP))


def pounds(pence: int) -> float:
    """Integer pence -> float GBP for the API boundary (exact: /100)."""
    return float(Decimal(pence) / 100)


def _ratio(num_pence: int, den_pence: int, dp: int = 3) -> float:
    if den_pence <= 0:
        return 0.0
    return float((Decimal(num_pence) / Decimal(den_pence)).quantize(Decimal(10) ** -dp))


# ---------------------------------------------------------------------------
# Loan maths (representative-APR annuity; pricing approximation, rounded to pence)
# ---------------------------------------------------------------------------


def monthly_repayment(amount_gbp: float, term_months: int, apr: float | None = None) -> float:
    apr = thresholds().apr if apr is None else apr
    if amount_gbp <= 0 or term_months <= 0:
        return 0.0
    r = apr / 12.0
    raw = amount_gbp / term_months if r == 0 else amount_gbp * r / (1 - (1 + r) ** -term_months)
    return pounds(to_pence(raw))


def max_amount_for_repayment(
    repayment_gbp: float, term_months: int, apr: float | None = None
) -> float:
    apr = thresholds().apr if apr is None else apr
    if repayment_gbp <= 0 or term_months <= 0:
        return 0.0
    r = apr / 12.0
    raw = (
        repayment_gbp * term_months if r == 0 else repayment_gbp * (1 - (1 + r) ** -term_months) / r
    )
    return pounds(to_pence(raw))


# ---------------------------------------------------------------------------
# Data-quality passes
# ---------------------------------------------------------------------------


def remove_duplicates(
    txns: list[CategorizedTransaction],
) -> tuple[list[CategorizedTransaction], list[str]]:
    """Drop exact re-posts: same account, date, description, amount and raw_type.
    First occurrence (lowest transaction_id) wins. Real feeds would use the bank's
    own FITID; this heuristic is a documented prototype simplification."""
    seen: dict[tuple, str] = {}
    kept, dropped = [], []
    for t in sorted(txns, key=lambda x: x.transaction_id):
        key = (t.account_id, t.date, t.description, to_pence(t.amount), t.raw_type)
        if key in seen:
            dropped.append(t.transaction_id)
        else:
            seen[key] = t.transaction_id
            kept.append(t)
    kept.sort(key=lambda x: (x.date, x.transaction_id))
    return kept, dropped


def match_refunds(
    txns: list[CategorizedTransaction],
) -> tuple[dict[str, Category], list[str], list[str]]:
    """Match each refund credit to an earlier debit whose description contains the
    refund's merchant stem (same account, within REFUND_MATCH_DAYS). Returns
    (refund_id -> category to net against, matched_ids, unmatched_ids)."""
    debits = [t for t in txns if t.amount < 0]
    matched: dict[str, Category] = {}
    matched_ids, unmatched = [], []
    for r in (t for t in txns if t.category == Category.refund and t.amount > 0):
        stem = r.description.upper().replace(" REFUND", "").split(" REFUND")[0].strip()
        stem = stem.split(" NOTE")[0].strip()
        best = None
        for d in debits:
            if d.account_id != r.account_id or d.date > r.date:
                continue
            if (
                _days_between(d.date, r.date) <= REFUND_MATCH_DAYS
                and stem
                and stem in d.description.upper()
            ):
                if best is None or d.date > best.date:
                    best = d
        if best is not None:
            matched[r.transaction_id] = best.category
            matched_ids += [r.transaction_id, best.transaction_id]
        else:
            unmatched.append(r.transaction_id)
    return matched, matched_ids, unmatched


def _days_between(d1: str, d2: str) -> int:
    from datetime import date

    return abs((date.fromisoformat(d2) - date.fromisoformat(d1)).days)


def observed_months(txns: Iterable[CategorizedTransaction]) -> tuple[list[str], bool]:
    """Distinct YYYY-MM months, plus whether the span has a gap (non-contiguous)."""
    months = sorted({t.date[:7] for t in txns})
    if len(months) < 2:
        return months, False
    y0, m0 = int(months[0][:4]), int(months[0][5:7])
    y1, m1 = int(months[-1][:4]), int(months[-1][5:7])
    span = (y1 - y0) * 12 + (m1 - m0) + 1
    return months, span != len(months)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    cat_txns: list[CategorizedTransaction], cfg: Thresholds | None = None
) -> AffordabilityMetrics:
    cfg = cfg or thresholds()
    evidence: dict[str, list[str]] = defaultdict(list)

    txns, dropped = remove_duplicates(list(cat_txns))
    if dropped:
        evidence["duplicates_removed"] = dropped

    months, gap = observed_months(txns)
    n_months = max(len(months), 1)

    transfers = [t for t in txns if t.category == Category.internal_transfer]
    evidence["internal_transfers_excluded"] = [t.transaction_id for t in transfers]
    transfer_net_p = sum(to_pence(t.amount) for t in transfers)
    transfer_gross_p = sum(abs(to_pence(t.amount)) for t in transfers)

    refund_target, refund_ids, unmatched_refunds = match_refunds(txns)
    if refund_ids:
        evidence["refunds_netted"] = refund_ids
    if unmatched_refunds:
        evidence["refunds_unmatched_excluded"] = unmatched_refunds

    # ---- income (eligible credits only; pence) ----
    inc_by_month: dict[str, int] = {m: 0 for m in months}
    benefits_total = 0
    income_desc: set[str] = set()
    for t in txns:
        if t.amount > 0 and t.category in ELIGIBLE_INCOME:
            p = to_pence(t.amount)
            inc_by_month[t.date[:7]] += p
            income_desc.add(t.description.strip().upper())
            evidence["eligible_income"].append(t.transaction_id)
            if t.category == Category.benefits:
                benefits_total += p
    monthly = list(inc_by_month.values())
    total_income = sum(monthly)
    mean_income_p = total_income // n_months if n_months else 0
    if total_income > 0 and n_months > 1:
        mean_d = Decimal(total_income) / n_months
        var = sum((Decimal(v) - mean_d) ** 2 for v in monthly) / n_months
        vol = float((var.sqrt() / mean_d).quantize(Decimal("0.001")))
    else:
        vol = 0.0
    income_months = sum(1 for v in monthly if v > 0)

    if mean_income_p <= 0:
        assessed_p = 0
    elif vol <= cfg.volatility_max:
        assessed_p = mean_income_p
    else:  # POL-004: conservative assessment of volatile income
        conservative = int(mean_income_p * (1 - min(vol, 0.5)))
        assessed_p = min(min(monthly), conservative)

    # ---- expenditure buckets (debits, abs pence, window totals) ----
    buckets: dict[str, int] = defaultdict(int)
    total_debits = 0
    distress_ids = []
    for t in txns:
        if t.category in EXCLUDED_MOVEMENT:
            continue
        if t.amount >= 0:
            continue
        p = -to_pence(t.amount)
        total_debits += p
        if t.category in BUCKET_ESSENTIAL:
            buckets["essential"] += p
            evidence["essential_spend"].append(t.transaction_id)
        elif t.category in BUCKET_DEBT:
            buckets["debt"] += p
            evidence["existing_debt_repayments"].append(t.transaction_id)
        elif t.category == Category.gambling:
            buckets["gambling"] += p
            evidence["gambling_spend"].append(t.transaction_id)
        elif t.category == Category.cash_withdrawal:
            buckets["cash"] += p
            evidence["cash_withdrawals"].append(t.transaction_id)
        elif t.category == Category.unknown:
            buckets["unknown"] += p
            evidence["unknown_spend"].append(t.transaction_id)
        elif t.category == Category.fees:
            buckets["fees"] += p
            distress_ids.append(t.transaction_id)
        elif t.category in BUCKET_DISCRETIONARY:
            buckets["discretionary"] += p
            evidence["discretionary_spend"].append(t.transaction_id)

    # refunds: net matched refunds against their category bucket (floor at zero)
    for rid, cat in refund_target.items():
        p = to_pence(next(t.amount for t in txns if t.transaction_id == rid))
        key = (
            "essential"
            if cat in BUCKET_ESSENTIAL
            else "debt"
            if cat in BUCKET_DEBT
            else "gambling"
            if cat == Category.gambling
            else "cash"
            if cat == Category.cash_withdrawal
            else "unknown"
            if cat == Category.unknown
            else "discretionary"
        )
        buckets[key] = max(buckets[key] - p, 0)
        total_debits = max(total_debits - p, 0)

    savings_out = sum(
        -to_pence(t.amount) for t in txns if t.category == Category.savings and t.amount < 0
    )

    def per_month(total_p: int) -> int:
        return int((Decimal(total_p) / n_months).to_integral_value(rounding=ROUND_HALF_UP))

    essential_m = per_month(buckets["essential"])
    debt_m = per_month(buckets["debt"])
    unknown_m = per_month(buckets["unknown"])
    disposable_p = assessed_p - essential_m - debt_m - unknown_m

    return AffordabilityMetrics(
        months_observed=len(months),
        coverage_gap=gap,
        n_transactions=len(txns),
        n_accounts=len({t.account_id for t in txns}),
        n_duplicates_removed=len(dropped),
        n_internal_transfer_txns=len(transfers),
        internal_transfer_net=pounds(transfer_net_p),
        internal_transfer_gross=pounds(transfer_gross_p),
        n_refunds_netted=len(refund_target),
        monthly_income_mean=pounds(mean_income_p),
        monthly_income_assessed=pounds(assessed_p),
        income_volatility=round(vol, 3),
        income_months=income_months,
        income_sources=len(income_desc),
        benefits_share=_ratio(benefits_total, total_income),
        essential_spend=pounds(essential_m),
        discretionary_spend=pounds(per_month(buckets["discretionary"])),
        existing_debt_repayments=pounds(debt_m),
        gambling_spend=pounds(per_month(buckets["gambling"])),
        cash_withdrawals=pounds(per_month(buckets["cash"])),
        unknown_spend=pounds(unknown_m),
        savings_transfers=pounds(per_month(savings_out)),
        unknown_share=_ratio(buckets["unknown"], total_debits),
        cash_share=_ratio(buckets["cash"], total_debits),
        essential_share=_ratio(essential_m, assessed_p),
        gambling_ratio=_ratio(per_month(buckets["gambling"]), assessed_p),
        dti_existing=_ratio(debt_m, assessed_p),
        disposable_income=pounds(disposable_p),
        distress_events=len(distress_ids),
        evidence={k: v for k, v in evidence.items() if v}
        | ({"distress_events": distress_ids} if distress_ids else {}),
    )
