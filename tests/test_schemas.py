"""Pydantic contract tests: what the schemas must accept and, more importantly,
what they must REJECT — invented fields, invalid dates, out-of-range confidence,
duplicate IDs, overrides without an outcome."""

import pytest
from pydantic import ValidationError

from schemas import (
    CategorizationBatchResult,
    CategorizedTransaction,
    Category,
    LendingDecision,
    Outcome,
    ReviewRequest,
    Transaction,
    WarningCode,
    make_warning,
)


def _tx(**kw):
    base = dict(
        transaction_id="TX-1",
        account_id="AC-1",
        date="2026-04-01",
        description="TESCO STORES",
        amount=-12.5,
        raw_type="DEB",
    )
    base.update(kw)
    return base


def test_transaction_valid():
    t = Transaction(**_tx())
    assert t.currency == "GBP" and t.amount == -12.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("date", "01/04/2026"),
        ("date", "2026-13-01"),
        ("amount", 0),
        ("amount", 12.345),
        ("amount", 2_000_000),
        ("currency", "EUR"),
        ("description", ""),
        ("transaction_id", ""),
    ],
)
def test_transaction_rejects_bad_values(field, value):
    with pytest.raises(ValidationError):
        Transaction(**_tx(**{field: value}))


def test_transaction_rejects_invented_fields():
    with pytest.raises(ValidationError):
        Transaction(**_tx(), sneaky_extra="x")


def test_categorized_confidence_bounds():
    with pytest.raises(ValidationError):
        CategorizedTransaction(**_tx(), category=Category.groceries, confidence=1.2)


def test_batch_result_rejects_duplicate_ids():
    with pytest.raises(ValidationError):
        CategorizationBatchResult(
            items=[
                {"transaction_id": "TX-1", "category": "groceries", "confidence": 0.9},
                {"transaction_id": "TX-1", "category": "dining", "confidence": 0.9},
            ]
        )


def test_batch_result_rejects_invalid_category_and_extra_fields():
    with pytest.raises(ValidationError):
        CategorizationBatchResult(
            items=[{"transaction_id": "TX-1", "category": "crypto", "confidence": 0.9}]
        )
    with pytest.raises(ValidationError):
        CategorizationBatchResult(
            items=[
                {
                    "transaction_id": "TX-1",
                    "category": "groceries",
                    "confidence": 0.9,
                    "amount": 999,
                }
            ]
        )  # the model may not emit amounts at all


def test_batch_result_detects_missing_and_invented_ids():
    r = CategorizationBatchResult(
        items=[
            {"transaction_id": "TX-1", "category": "groceries", "confidence": 0.9},
            {"transaction_id": "TX-99", "category": "dining", "confidence": 0.8},
        ]
    )
    missing, invented = r.validate_against_inputs({"TX-1", "TX-2"})
    assert missing == {"TX-2"} and invented == {"TX-99"}


def test_decision_confidence_validated():
    with pytest.raises(ValidationError):
        LendingDecision(
            applicant_id="x",
            outcome=Outcome.approve,
            confidence=2.0,
            requested_amount=1,
            term_months=12,
            monthly_repayment=1,
            max_affordable_amount=1,
            disposable_after_repayment=1,
            dti_including_new=0.1,
        )


def test_warning_severity_mapping():
    assert make_warning(WarningCode.insufficient_history, "x").severity == "guardrail"
    assert make_warning(WarningCode.gambling_high, "x").severity == "review"
    assert make_warning(WarningCode.refunds_netted, "x").severity == "info"


def test_review_override_requires_outcome():
    with pytest.raises(ValidationError):
        ReviewRequest(reviewer="r", action="override", reason="long enough reason")
    ok = ReviewRequest(
        reviewer="r", action="override", final_outcome="decline", reason="long enough reason"
    )
    assert ok.final_outcome == Outcome.decline


def test_review_requires_substantive_reason():
    with pytest.raises(ValidationError):
        ReviewRequest(reviewer="r", action="uphold", reason="short")


# ---------------------------------------------------------------------------
# sign-consistency: a category can never contradict the transaction's sign
# ---------------------------------------------------------------------------


def test_sign_consistency_rejects_credit_only_category_on_debit():
    with pytest.raises(ValidationError):
        CategorizedTransaction(**_tx(amount=-900.0), category=Category.income, confidence=0.9)


def test_sign_consistency_rejects_debit_only_category_on_credit():
    with pytest.raises(ValidationError):
        CategorizedTransaction(**_tx(amount=900.0), category=Category.rent_mortgage, confidence=0.9)


def test_sign_consistency_allows_both_sign_categories():
    for amt in (250.0, -250.0):
        t = CategorizedTransaction(
            **_tx(amount=amt), category=Category.internal_transfer, confidence=0.9
        )
        assert t.category == Category.internal_transfer
