"""Shared fixtures. ALL tests run offline and deterministically: no network, no
API keys, fixed seeds — LLM behaviour is exercised through FakeLLM (tests/fakes.py).
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# --------------------------------------------------------------------------
# Hermetic configuration.
#
# Every threshold and LLM setting is env-overridable by design, which means an
# ambient shell variable can silently change what the offline suite measures.
# That is not hypothetical: on 2026-08-02 an exported LLM_BATCH_SIZE=10, left over
# from a live Ollama evaluation, split a 40-item scripted FakeLLM response into
# four batches. Thirty items fell back to rules, so llm_output_invalid fired at
# the CATEGORISATION stage instead of llm_unavailable at the RATIONALE stage, and
# a guardrail-precedence assertion failed on a machine where nothing was wrong
# with the code. CI never saw it because CI starts from a clean environment.
#
# Tests therefore pin the DEFAULTS in src/config.py. To exercise an override, use
# monkeypatch.setenv inside the test (auto-undone) — never the ambient shell.
# --------------------------------------------------------------------------
_CONFIG_ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_BATCH_SIZE",
    "LLM_TIMEOUT_S",
    "LLM_MAX_RETRIES",
    "LLM_MAX_TOKENS",
    "LLM_TEMPERATURE",
    "LLM_SEED",
    "OLLAMA_HOST",
    "EMBEDDINGS",
    "RETRIEVAL_K",
    "MIN_SCORE_TFIDF",
    "MIN_SCORE_MINILM",
    "APR",
    "BUFFER_GBP",
    "DTI_REFER",
    "DTI_MAX",
    "GAMBLING_REFER",
    "VOLATILITY_MAX",
    "DISTRESS_MAX",
    "MIN_MONTHS",
    "MIN_INCOME_MONTHS",
    "MIN_TRANSACTIONS",
    "MAX_TRANSACTIONS",
    "MAX_UNKNOWN_SHARE",
    "MAX_CASH_SHARE",
    "BENEFITS_SHARE_REVIEW",
    "ESSENTIAL_SHARE_HIGH",
    "REDUCED_OFFER_MIN_GBP",
    "REDUCED_OFFER_MIN_FRACTION",
    "TRANSFER_IMBALANCE_TOLERANCE",
    "TRANSFER_IMBALANCE_FLOOR_GBP",
)

for _var in _CONFIG_ENV_VARS:
    os.environ.pop(_var, None)

from dataio import DEV_DIR, EVAL_DIR, load_dataset, load_ground_truth  # noqa: E402
from retriever import PolicyRetriever  # noqa: E402


def _ensure(which: str, d: pathlib.Path) -> None:
    if not (d / "applicants.csv").exists():
        subprocess.run(
            [sys.executable, str(ROOT / "data" / "generate_data.py"), "--set", which], check=True
        )


_ensure("dev", DEV_DIR)
_ensure("eval", EVAL_DIR)


@pytest.fixture(scope="session")
def retriever() -> PolicyRetriever:
    return PolicyRetriever()  # tfidf backend: offline + deterministic


@pytest.fixture(scope="session")
def dev_data():
    return load_dataset(DEV_DIR)


@pytest.fixture(scope="session")
def eval_data():
    return load_dataset(EVAL_DIR)


@pytest.fixture(scope="session")
def eval_truth():
    return load_ground_truth(EVAL_DIR)


_SEQ = {"n": 0}


def make_txn(
    description: str = "TESCO STORES",
    amount: float = -25.0,
    date: str = "2026-04-10",
    account: str = "AC-900-CUR",
    raw_type: str = "DEB",
    tid: str | None = None,
) -> dict:
    _SEQ["n"] += 1
    return {
        "transaction_id": tid or f"TX-900-{_SEQ['n']:05d}",
        "account_id": account,
        "date": date,
        "description": description,
        "amount": amount,
        "raw_type": raw_type,
    }


@pytest.fixture()
def mk_txn():
    return make_txn
