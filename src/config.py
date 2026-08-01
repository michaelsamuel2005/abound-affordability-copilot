"""Central configuration: decision thresholds, retrieval settings and versions.

Every policy threshold lives here (not scattered through code) and can be
overridden by environment variable without a code change — mirroring how a real
lender would want credit policy parameters versioned and configurable.

The threshold VALUES are invented for this prototype. They are loosely informed
by public FCA CONC 5.2A affordability principles and common industry heuristics
(e.g. debt-to-income bands), but they are NOT any real lender's policy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Versions — stamped into every decision, log line and report for auditability.
# ---------------------------------------------------------------------------
ENGINE_VERSION = "2.0.0"  # deterministic affordability + decision rules
POLICY_VERSION = "2026-07.2"  # version of the policy corpus in /policy
DATASET_VERSION = "2026-07.r2"  # synthetic dataset generator version


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Thresholds:
    """Decision-policy parameters (units: GBP for money, ratios 0-1)."""

    # -- data sufficiency (guardrails) --
    min_months: int = field(default_factory=lambda: _i("MIN_MONTHS", 3))
    min_transactions: int = field(default_factory=lambda: _i("MIN_TRANSACTIONS", 40))
    min_income_months: int = field(default_factory=lambda: _i("MIN_INCOME_MONTHS", 2))
    max_unknown_share: float = field(default_factory=lambda: _f("MAX_UNKNOWN_SHARE", 0.10))
    max_cash_share: float = field(default_factory=lambda: _f("MAX_CASH_SHARE", 0.25))
    # classification-plausibility guardrails (DQ-007)
    transfer_imbalance_tolerance: float = field(
        default_factory=lambda: _f("TRANSFER_IMBALANCE_TOLERANCE", 0.20)
    )
    transfer_imbalance_floor_gbp: float = field(
        default_factory=lambda: _f("TRANSFER_IMBALANCE_FLOOR_GBP", 100.0)
    )

    # -- affordability rules --
    buffer_gbp: float = field(default_factory=lambda: _f("BUFFER_GBP", 150.0))
    dti_refer: float = field(default_factory=lambda: _f("DTI_REFER", 0.40))
    dti_max: float = field(default_factory=lambda: _f("DTI_MAX", 0.45))
    gambling_refer: float = field(default_factory=lambda: _f("GAMBLING_REFER", 0.10))
    volatility_max: float = field(default_factory=lambda: _f("VOLATILITY_MAX", 0.35))
    distress_max: int = field(default_factory=lambda: _i("DISTRESS_MAX", 2))
    benefits_share_review: float = field(default_factory=lambda: _f("BENEFITS_SHARE_REVIEW", 0.50))
    essential_share_high: float = field(default_factory=lambda: _f("ESSENTIAL_SHARE_HIGH", 0.60))

    # -- reduced-offer path when the requested amount fails the buffer test --
    reduced_offer_min_gbp: float = field(default_factory=lambda: _f("REDUCED_OFFER_MIN_GBP", 500.0))
    reduced_offer_min_fraction: float = field(
        default_factory=lambda: _f("REDUCED_OFFER_MIN_FRACTION", 0.50)
    )

    # -- loan pricing (representative, for repayment maths only) --
    apr: float = field(default_factory=lambda: _f("APR", 0.249))


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = field(default_factory=lambda: _i("RETRIEVAL_K", 4))
    # below this cosine score a chunk is treated as not relevant
    min_score_tfidf: float = field(default_factory=lambda: _f("MIN_SCORE_TFIDF", 0.05))
    min_score_minilm: float = field(default_factory=lambda: _f("MIN_SCORE_MINILM", 0.25))


@dataclass(frozen=True)
class LLMConfig:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "").lower().strip())
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    temperature: float = field(default_factory=lambda: _f("LLM_TEMPERATURE", 0.0))
    max_output_tokens: int = field(default_factory=lambda: _i("LLM_MAX_TOKENS", 1024))
    timeout_s: float = field(default_factory=lambda: _f("LLM_TIMEOUT_S", 60.0))
    max_retries: int = field(default_factory=lambda: _i("LLM_MAX_RETRIES", 2))
    seed: int = field(default_factory=lambda: _i("LLM_SEED", 42))
    batch_size: int = field(default_factory=lambda: _i("LLM_BATCH_SIZE", 40))


@dataclass(frozen=True)
class APIConfig:
    max_transactions: int = field(default_factory=lambda: _i("MAX_TRANSACTIONS", 5000))


def thresholds() -> Thresholds:
    return Thresholds()


def retrieval_config() -> RetrievalConfig:
    return RetrievalConfig()


def llm_config() -> LLMConfig:
    return LLMConfig()


def api_config() -> APIConfig:
    return APIConfig()
