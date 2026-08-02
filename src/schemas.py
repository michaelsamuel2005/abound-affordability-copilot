"""Pydantic schemas — the structured-output contracts of the whole pipeline.

Three roles:
  1.  API contracts (Transaction in, LendingDecision out) — everything entering
      or leaving the service is validated.
  2.  LLM contracts (CategoryAssignment / CategorizationBatchResult) — the model
      may only answer inside this schema; anything else is rejected and repaired.
      `extra="forbid"` means invented fields fail validation rather than passing
      silently.
  3.  Audit contracts (WarningFlag, PolicyCitation, ReviewRecord) — warnings are
      fixed machine-readable codes, citations carry the verbatim policy text they
      quote, and every reviewer action is recorded.

Money convention: amounts cross the API boundary as JSON numbers in GBP with at
most 2 decimal places (Open-Banking style). All *arithmetic* happens in integer
pence inside `affordability.py`; floats are display/boundary only.
Sign convention: positive = credit (money in), negative = debit (money out).
"""

from __future__ import annotations

from datetime import date as _date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Category taxonomy (21 categories, mutually exclusive, exactly one per txn)
# ---------------------------------------------------------------------------


class Category(str, Enum):
    # -- inflows --
    income = "income"  # salary / self-employed / gig earnings
    benefits = "benefits"  # state benefits (e.g. Universal Credit)
    refund = "refund"  # merchant refund / reversal of a purchase
    # -- money movement (excluded from income AND spend) --
    internal_transfer = "internal_transfer"  # between the customer's own accounts
    savings = "savings"  # transfer into savings/pots
    # -- essential outflows --
    rent_mortgage = "rent_mortgage"
    utilities = "utilities"
    council_tax = "council_tax"
    groceries = "groceries"
    transport = "transport"
    insurance = "insurance"
    # -- existing credit commitments --
    loan_repayment = "loan_repayment"
    bnpl = "bnpl"  # buy-now-pay-later instalments
    # -- risk indicators --
    gambling = "gambling"
    fees = "fees"  # overdraft / returned-DD / unpaid-item fees
    # -- discretionary outflows --
    subscriptions = "subscriptions"
    dining = "dining"
    shopping = "shopping"
    entertainment = "entertainment"
    cash_withdrawal = "cash_withdrawal"  # ATM — spend purpose unverifiable
    # -- abstention --
    unknown = "unknown"  # cannot determine; triggers conservative handling


ESSENTIAL = {
    Category.rent_mortgage,
    Category.utilities,
    Category.council_tax,
    Category.groceries,
    Category.transport,
    Category.insurance,
}
DEBT = {Category.loan_repayment, Category.bnpl}
DISCRETIONARY = {
    Category.subscriptions,
    Category.dining,
    Category.shopping,
    Category.entertainment,
    Category.cash_withdrawal,
}
ELIGIBLE_INCOME = {Category.income, Category.benefits}
# excluded from both income and expenditure aggregates (money movement, not consumption)
NEUTRAL = {Category.internal_transfer, Category.savings}

CATEGORY_VALUES = {c.value for c in Category}

# sign-consistency: these categories are only meaningful on one side of the
# account. A credit labelled `rent_mortgage`, or a debit labelled `income`,
# is structurally invalid — enforced at CategorizedTransaction level so no
# producer (rules, LLM, or a human correction) can smuggle one through.
CREDIT_ONLY = {Category.income, Category.benefits, Category.refund}
DEBIT_ONLY = ESSENTIAL | DEBT | DISCRETIONARY | {Category.gambling, Category.fees}


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


class Transaction(BaseModel):
    """One Open-Banking-style transaction. IDs are mandatory: every downstream
    figure must be traceable back to specific transaction IDs."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1, max_length=64)
    account_id: str = Field(min_length=1, max_length=64)
    date: str = Field(description="Booking date, ISO YYYY-MM-DD")
    description: str = Field(min_length=1, max_length=200)
    amount: float = Field(description="GBP; + credit, - debit; max 2dp")
    currency: str = "GBP"
    raw_type: str = Field(
        default="", max_length=16, description="Bank-provided hint, e.g. BGC/DD/DEB/FPI/TFR/FEE/ATM"
    )

    @field_validator("date")
    @classmethod
    def _iso_date(cls, v: str) -> str:
        _date.fromisoformat(v)  # raises ValueError if malformed
        return v

    @field_validator("amount")
    @classmethod
    def _nonzero_2dp(cls, v: float) -> float:
        if v == 0:
            raise ValueError("amount must be non-zero")
        if abs(round(v * 100) - v * 100) > 1e-6:
            raise ValueError("amount must have at most 2 decimal places")
        if abs(v) > 1_000_000:
            raise ValueError("amount out of plausible range")
        return v

    @field_validator("currency")
    @classmethod
    def _gbp_only(cls, v: str) -> str:
        if v != "GBP":
            raise ValueError("only GBP is supported in this prototype")
        return v


class CategorizedTransaction(Transaction):
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["rules", "llm", "llm_repair", "rule_fallback", "human"] = "rules"
    is_essential: bool = False

    @model_validator(mode="after")
    def _sign_consistent(self) -> CategorizedTransaction:
        if self.amount > 0 and self.category in DEBIT_ONLY:
            raise ValueError(
                f"category '{self.category.value}' is debit-only but the amount is a credit"
            )
        if self.amount < 0 and self.category in CREDIT_ONLY:
            raise ValueError(
                f"category '{self.category.value}' is credit-only but the amount is a debit"
            )
        return self


# ---------------------------------------------------------------------------
# LLM structured-output contract for categorisation
# ---------------------------------------------------------------------------


class CategoryAssignment(BaseModel):
    """One item of the LLM's answer. extra='forbid' → invented fields rejected."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)


class CategorizationBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[CategoryAssignment]

    @model_validator(mode="after")
    def _unique_ids(self) -> CategorizationBatchResult:
        ids = [i.transaction_id for i in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate transaction_id in categorisation output")
        return self

    def validate_against_inputs(self, input_ids: set[str]) -> tuple[set[str], set[str]]:
        """Return (missing_ids, invented_ids) relative to the transactions sent."""
        got = {i.transaction_id for i in self.items}
        return input_ids - got, got - input_ids


# ---------------------------------------------------------------------------
# Affordability metrics (computed deterministically; GBP at this boundary)
# ---------------------------------------------------------------------------


class AffordabilityMetrics(BaseModel):
    # data window & quality
    months_observed: int
    coverage_gap: bool = False  # non-contiguous months in the window
    n_transactions: int
    n_accounts: int = 1  # distinct connected accounts observed in the window
    n_duplicates_removed: int = 0
    n_internal_transfer_txns: int = 0
    internal_transfer_net: float = 0.0  # signed sum of transfer legs (should be ~0)
    internal_transfer_gross: float = 0.0  # sum of |transfer legs|
    n_refunds_netted: int = 0

    # income
    monthly_income_mean: float  # mean monthly eligible income (income+benefits)
    monthly_income_assessed: float  # conservative figure actually used
    income_volatility: float  # coefficient of variation of monthly eligible income
    income_months: int  # months in which eligible income was seen
    income_sources: int  # distinct credit descriptions among eligible income
    benefits_share: float  # benefits / eligible income

    # expenditure (monthly averages)
    essential_spend: float
    discretionary_spend: float
    existing_debt_repayments: float
    gambling_spend: float
    cash_withdrawals: float
    unknown_spend: float  # debits categorised `unknown` (conservative: reduces disposable)
    savings_transfers: float  # informational; excluded from spend

    # shares & ratios
    unknown_share: float  # |unknown debit value| / |total debit value|
    cash_share: float
    essential_share: float  # essential spend / assessed income (0 if no income)
    gambling_ratio: float  # gambling spend / assessed income
    dti_existing: float  # existing debt repayments / assessed income

    disposable_income: float  # assessed income - essentials - debt - unknown spend
    distress_events: int

    # audit trail: metric name -> supporting transaction IDs
    evidence: dict[str, list[str]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Warnings / guardrails (fixed machine-readable codes, never free text)
# ---------------------------------------------------------------------------


class WarningCode(str, Enum):
    # guardrails — data cannot support an automated recommendation → forced refer
    insufficient_history = "insufficient_history"
    low_transaction_count = "low_transaction_count"
    no_recognisable_income = "no_recognisable_income"
    coverage_gap = "coverage_gap"
    high_unknown_share = "high_unknown_share"
    high_cash_usage = "high_cash_usage"
    retrieval_empty = "retrieval_empty"
    llm_output_invalid = "llm_output_invalid"
    llm_unavailable = "llm_unavailable"
    transfer_imbalance = "transfer_imbalance"  # transfer legs don't balance
    no_essential_spend = "no_essential_spend"  # implausible zero essentials
    # review triggers — assessable, but a human must look
    income_volatility_high = "income_volatility_high"
    gambling_high = "gambling_high"
    distress_events = "distress_events"
    vulnerability_indicators = "vulnerability_indicators"
    dti_borderline = "dti_borderline"
    buffer_breach_reduced_offer = "buffer_breach_reduced_offer"
    # informational — data treatments applied, no action needed
    duplicates_removed = "duplicates_removed"
    internal_transfers_netted = "internal_transfers_netted"
    refunds_netted = "refunds_netted"
    llm_output_repaired = "llm_output_repaired"
    rationale_rejected = "rationale_rejected"


GUARDRAIL_CODES = {
    WarningCode.insufficient_history,
    WarningCode.low_transaction_count,
    WarningCode.no_recognisable_income,
    WarningCode.coverage_gap,
    WarningCode.high_unknown_share,
    WarningCode.high_cash_usage,
    WarningCode.retrieval_empty,
    WarningCode.llm_output_invalid,
    WarningCode.llm_unavailable,
    WarningCode.transfer_imbalance,
    WarningCode.no_essential_spend,
}
REVIEW_CODES = {
    WarningCode.income_volatility_high,
    WarningCode.gambling_high,
    WarningCode.distress_events,
    WarningCode.vulnerability_indicators,
    WarningCode.dti_borderline,
    WarningCode.buffer_breach_reduced_offer,
}


class WarningFlag(BaseModel):
    code: WarningCode
    severity: Literal["guardrail", "review", "info"]
    message: str


def make_warning(code: WarningCode, message: str) -> WarningFlag:
    if code in GUARDRAIL_CODES:
        sev = "guardrail"
    elif code in REVIEW_CODES:
        sev = "review"
    else:
        sev = "info"
    return WarningFlag(code=code, severity=sev, message=message)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


class Outcome(str, Enum):
    approve = "approve"
    refer = "refer"
    decline = "decline"


class PolicyCitation(BaseModel):
    policy_id: str  # e.g. POL-005
    doc_id: str  # e.g. lending_policy
    title: str
    version: str
    quote: str = Field(max_length=400)  # verbatim excerpt of the section text
    score: float | None = None  # retrieval similarity if retrieved


class LendingDecision(BaseModel):
    applicant_id: str
    outcome: Outcome
    confidence: float = Field(ge=0.0, le=1.0)

    requested_amount: float
    term_months: int
    monthly_repayment: float
    max_affordable_amount: float
    disposable_after_repayment: float
    dti_including_new: float

    reasons: list[str] = Field(default_factory=list)  # deterministic reason sentences
    rationale: str = ""  # narrative (template or LLM, validated)
    warnings: list[WarningFlag] = Field(default_factory=list)
    guardrail: WarningCode | None = None  # first guardrail-severity warning
    policy_citations: list[PolicyCitation] = Field(default_factory=list)

    human_review_required: bool = True  # the copilot recommends; it never decides
    review_priority: Literal["standard", "high"] = "standard"
    versions: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Human review (HITL)
# ---------------------------------------------------------------------------


class ReviewActionType(str, Enum):
    uphold = "uphold"
    override = "override"


class CategoryCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction_id: str
    category: Category


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer: str = Field(min_length=1, max_length=80)
    action: ReviewActionType
    final_outcome: Outcome | None = None
    reason: str = Field(min_length=10, max_length=1000)  # an override always needs a reason
    category_corrections: list[CategoryCorrection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _override_needs_outcome(self) -> ReviewRequest:
        if self.action == ReviewActionType.override and self.final_outcome is None:
            raise ValueError("final_outcome is required when action is 'override'")
        return self


class ReviewRecord(BaseModel):
    review_id: str
    assessment_id: str
    reviewer: str
    action: ReviewActionType
    original_outcome: Outcome
    final_outcome: Outcome
    reason: str
    n_corrections: int = 0
    corrections: list[CategoryCorrection] = Field(default_factory=list)  # exact edits, persisted
    recomputed: bool = False  # were metrics/decision recomputed after corrections?
    recomputed_outcome: Outcome | None = None
    recomputed_metrics: dict | None = None  # full post-correction metrics, persisted
    created_at: str = ""
