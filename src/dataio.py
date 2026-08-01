"""Dataset loading. Strips ground-truth columns so the pipeline can never see
labels: `true_category`, `expected_*` and `balance_after` are visible only to the
evaluation harness, which loads them separately via `load_ground_truth`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEV_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"

PIPELINE_TX_FIELDS = ["transaction_id", "account_id", "date", "description", "amount", "raw_type"]


def load_dataset(data_dir: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """Return (applicants, transactions-by-applicant) with ONLY pipeline-visible
    fields — the model and engine never see ground-truth labels."""
    apps = pd.read_csv(data_dir / "applicants.csv").to_dict("records")
    tx = pd.read_csv(data_dir / "transactions.csv")
    tx_by: dict[str, list[dict]] = {}
    for aid, g in tx.groupby("applicant_id"):
        tx_by[str(aid)] = g[PIPELINE_TX_FIELDS].to_dict("records")
    for a in apps:
        for key in ("expected_warnings", "expected_policy_ids"):
            v = a.get(key)
            v = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
            a[key] = [s for s in v.split("|") if s]
    return apps, tx_by


def load_ground_truth(data_dir: Path) -> dict[str, str]:
    """transaction_id -> true category. Evaluation harness only."""
    tx = pd.read_csv(data_dir / "transactions.csv")
    return dict(zip(tx["transaction_id"].astype(str), tx["true_category"].astype(str), strict=True))
