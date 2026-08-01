"""Shared fixtures. ALL tests run offline and deterministically: no network, no
API keys, fixed seeds — LLM behaviour is exercised through FakeLLM (tests/fakes.py).
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
