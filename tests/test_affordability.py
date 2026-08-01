"""Affordability-engine tests: exact pence arithmetic, every data-quality pass
(duplicates, transfers, refunds, unknowns), the conservative-income formula, and
the evidence trail (every aggregate reproducible from its transaction IDs)."""

from decimal import Decimal

from conftest import make_txn

from affordability import (
    compute_metrics,
    max_amount_for_repayment,
    monthly_repayment,
    pounds,
    remove_duplicates,
    to_pence,
)
from categorize import categorize_rules


def _cats(txns):
    return categorize_rules(txns)[0]


def _month_range(n=3):
    return [f"2026-{m:02d}" for m in range(4, 4 + n)]


def _baseline(months=3, income=3000.0, rent=900.0):
    """A clean applicant: salary + rent + groceries each month."""
    txns = []
    for m in _month_range(months):
        txns.append(
            make_txn(description="ACME LTD SALARY", amount=income, date=f"{m}-28", raw_type="BGC")
        )
        txns.append(
            make_txn(description="RENT PROPERTYCO", amount=-rent, date=f"{m}-01", raw_type="DD")
        )
        txns.append(make_txn(description="TESCO STORES", amount=-200.0, date=f"{m}-10"))
    return txns


# ---------------------------------------------------------------------------
# money primitives
# ---------------------------------------------------------------------------


def test_to_pence_is_exact():
    assert to_pence(12.34) == 1234
    assert to_pence("0.1") == 10
    assert to_pence(0.1 + 0.2) == 30  # float noise must not leak in
    assert to_pence(2.675) == 268  # HALF_UP, unlike float round()
    assert pounds(1234) == 12.34


def test_repayment_roundtrip_and_edges():
    rep = monthly_repayment(6000, 24)
    assert 250 < rep < 350
    assert abs(max_amount_for_repayment(rep, 24) - 6000) < 1.0
    assert monthly_repayment(0, 24) == 0.0
    assert monthly_repayment(6000, 24, apr=0.0) == 250.0
    assert max_amount_for_repayment(0, 24) == 0.0


# ---------------------------------------------------------------------------
# data-quality passes
# ---------------------------------------------------------------------------


def test_duplicates_removed_only_when_all_fields_match():
    a = make_txn(description="RENT PROPERTYCO", amount=-900.0, date="2026-04-01", raw_type="DD")
    dup = dict(a, transaction_id="TX-DUP-1")
    similar = dict(a, transaction_id="TX-DUP-2", amount=-900.01)  # differs by 1p
    kept, dropped = remove_duplicates(_cats([a, dup, similar]))
    assert dropped == ["TX-DUP-1"] and len(kept) == 2


def test_duplicate_feeds_metrics_and_evidence():
    txns = _baseline()
    dup = dict(txns[1], transaction_id="TX-REPOST")
    m = compute_metrics(_cats(txns + [dup]))
    assert m.n_duplicates_removed == 1
    assert "TX-REPOST" in m.evidence["duplicates_removed"]
    assert m.essential_spend == 1100.0  # rent counted once: 900 + 200 groceries


def test_internal_transfers_excluded_from_income_and_spend():
    txns = _baseline()
    for mth in _month_range():
        txns.append(
            make_txn(
                description="TRANSFER TO SAVINGS ACCOUNT",
                amount=-400.0,
                date=f"{mth}-26",
                raw_type="TFR",
            )
        )
        txns.append(
            make_txn(
                description="TRANSFER FROM CURRENT ACCOUNT",
                amount=400.0,
                date=f"{mth}-26",
                account="AC-900-SAV",
                raw_type="TFR",
            )
        )
    m = compute_metrics(_cats(txns))
    assert m.n_internal_transfer_txns == 6
    assert m.monthly_income_mean == 3000.0  # credits not inflated
    assert m.essential_spend == 1100.0  # debits not inflated


def test_matched_refund_nets_against_original_category():
    txns = _baseline()
    txns.append(make_txn(description="AMAZON.CO.UK", amount=-90.0, date="2026-04-12"))
    txns.append(
        make_txn(description="AMAZON.CO.UK REFUND", amount=90.0, date="2026-04-16", raw_type="FPI")
    )
    m = compute_metrics(_cats(txns))
    assert m.n_refunds_netted == 1
    assert m.discretionary_spend == 0.0  # purchase fully netted
    assert m.monthly_income_mean == 3000.0  # refund never income


def test_unmatched_refund_excluded_entirely():
    txns = _baseline()
    txns.append(
        make_txn(description="MYSTERY SHOP REFUND", amount=55.0, date="2026-04-16", raw_type="FPI")
    )
    m = compute_metrics(_cats(txns))
    assert m.n_refunds_netted == 0
    assert m.monthly_income_mean == 3000.0
    assert "refunds_unmatched_excluded" in m.evidence


def test_unknown_debits_reduce_disposable_conservatively():
    plain = compute_metrics(_cats(_baseline()))
    txns = _baseline()
    for mth in _month_range():
        txns.append(make_txn(description="SQ *MYSTERY VENDOR", amount=-120.0, date=f"{mth}-15"))
    m = compute_metrics(_cats(txns))
    assert m.unknown_spend == 120.0
    assert m.disposable_income == plain.disposable_income - 120.0
    assert m.unknown_share > 0


# ---------------------------------------------------------------------------
# income assessment
# ---------------------------------------------------------------------------


def test_stable_income_uses_mean():
    m = compute_metrics(_cats(_baseline()))
    assert m.monthly_income_assessed == m.monthly_income_mean == 3000.0
    assert m.income_volatility < 0.01 and m.income_months == 3


def test_volatile_income_assessed_conservatively():
    txns = []
    for mth, amt in zip(_month_range(3), [1000.0, 3000.0, 5000.0], strict=True):
        txns.append(
            make_txn(description="UPWORK PAYOUT", amount=amt, date=f"{mth}-15", raw_type="FPI")
        )
        txns.append(
            make_txn(description="RENT PROPERTYCO", amount=-800.0, date=f"{mth}-01", raw_type="DD")
        )
    m = compute_metrics(_cats(txns))
    # mean 3000, population std ~1633 -> cv ~0.544 -> capped at 0.5
    assert m.income_volatility > 0.5
    expected = min(1000.0, 3000.0 * 0.5)  # min(worst month, mean*(1-min(cv,.5)))
    assert m.monthly_income_assessed == expected


def test_no_eligible_income():
    txns = [
        make_txn(
            description="PAYPAL TRANSFER J SMITH", amount=2000.0, date="2026-04-05", raw_type="FPI"
        ),
        make_txn(description="RENT PROPERTYCO", amount=-700.0, date="2026-04-01", raw_type="DD"),
    ]
    m = compute_metrics(_cats(txns))
    assert m.monthly_income_assessed == 0.0 and m.income_months == 0
    assert m.disposable_income < 0  # spend with no recognised income


def test_benefits_share():
    txns = _baseline()
    for mth in _month_range():
        txns.append(
            make_txn(description="DWP UC PAYMENT", amount=1000.0, date=f"{mth}-01", raw_type="FPI")
        )
    m = compute_metrics(_cats(txns))
    assert m.monthly_income_mean == 4000.0
    assert m.benefits_share == 0.25


# ---------------------------------------------------------------------------
# window / coverage
# ---------------------------------------------------------------------------


def test_coverage_gap_detected():
    txns = [
        make_txn(date="2026-01-10"),
        make_txn(date="2026-02-10"),
        make_txn(date="2026-04-10"),
    ]  # March missing
    m = compute_metrics(_cats(txns))
    assert m.coverage_gap is True and m.months_observed == 3


def test_single_month_no_gap_no_volatility():
    txns = [
        make_txn(description="ACME LTD SALARY", amount=2000.0, date="2026-04-28", raw_type="BGC")
    ]
    m = compute_metrics(_cats(txns))
    assert m.months_observed == 1 and not m.coverage_gap
    assert m.income_volatility == 0.0


def test_cash_share():
    txns = _baseline(months=1)
    txns.append(
        make_txn(
            description="CASH WDL LLOYDS ATM", amount=-1100.0, date="2026-04-11", raw_type="ATM"
        )
    )
    m = compute_metrics(_cats(txns))
    # debits: 900 rent + 200 groceries + 1100 cash = 2200 -> cash 50%
    assert m.cash_share == 0.5 and m.cash_withdrawals == 1100.0


# ---------------------------------------------------------------------------
# evidence trail: every aggregate reproducible from its transaction IDs
# ---------------------------------------------------------------------------


def test_essential_spend_reproducible_from_evidence():
    txns = _baseline()
    cats = _cats(txns)
    m = compute_metrics(cats)
    by_id = {c.transaction_id: c for c in cats}
    total = sum(-to_pence(by_id[tid].amount) for tid in m.evidence["essential_spend"])
    assert pounds(int(Decimal(total) / m.months_observed)) == m.essential_spend


def test_income_reproducible_from_evidence():
    txns = _baseline()
    cats = _cats(txns)
    m = compute_metrics(cats)
    by_id = {c.transaction_id: c for c in cats}
    total = sum(to_pence(by_id[tid].amount) for tid in m.evidence["eligible_income"])
    assert pounds(total // m.months_observed) == m.monthly_income_mean
