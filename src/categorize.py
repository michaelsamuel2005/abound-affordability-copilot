"""Transaction categorisation. Two paths, one Pydantic contract.

Deterministic path (default): ordered keyword rules over description + raw_type.
Sign is NOT treated as category evidence for inflows — refunds, internal
transfers and benefits are money-in but must never count as income, and an
unrecognised inflow is `unknown` (conservative), not income. This asymmetry
exists because the costliest categorisation error in lending is inflating
income.

LLM path (when a provider is configured): batched structured-output
classification with a validate-and-repair loop:

    prompt (v3, injection-hardened) -> JSON -> Pydantic CategorizationBatchResult
      -> reject invented/duplicate/missing transaction_ids
      -> one repair attempt quoting the exact validation error
      -> per-item fallback to the deterministic rules (never an invalid value)
      -> sign-consistency: a credit can never carry a debit-only category (or
         vice versa) — schema-enforced, violating items fall back to rules

The LLM proposes categories and confidences ONLY. is_essential comes from our
own category mapping, amounts/dates/IDs are never taken from model output, and
every anomaly is counted in `meta` so the evaluation harness can report
structured-output success, repair and fallback rates.
"""

from __future__ import annotations

from pydantic import ValidationError

import prompts
from config import llm_config
from llm import BaseLLM, CallStats, LLMUnavailable, extract_json
from schemas import ESSENTIAL, CategorizationBatchResult, CategorizedTransaction, Category

# ---------------------------------------------------------------------------
# Deterministic keyword rules (ordered: first hit wins)
# ---------------------------------------------------------------------------

CREDIT_RULES: list[tuple[Category, list[str]]] = [
    (Category.refund, ["refund", "reversal", "rfnd"]),
    (Category.benefits, ["dwp", "universal credit"]),
    (Category.savings, ["monzo pot"]),
    (Category.income, ["salary", "payroll", "wages", "rider pay", "partner payment", "payout"]),
]

DEBIT_RULES: list[tuple[Category, list[str]]] = [
    (Category.fees, ["overdraft fee", "returned dd", "unpaid item"]),
    (Category.cash_withdrawal, ["atm", "cash wdl"]),
    (Category.savings, ["monzo pot"]),
    (Category.gambling, ["bet365", "paddypower", "skybet", "ladbrokes", "betfair", "gambl"]),
    (Category.council_tax, ["council tax"]),
    (Category.bnpl, ["klarna", "clearpay", "laybuy"]),
    (Category.subscriptions, ["netflix", "spotify", "amazon prime", "disney", "subscription"]),
    (Category.rent_mortgage, ["rent", "mortgage"]),
    (
        Category.utilities,
        [
            "british gas",
            "thames water",
            "edf",
            "octopus",
            "energy",
            "broadband",
            "vodafone",
            "water",
        ],
    ),
    (Category.insurance, ["insurance", "aviva", "admiral", "direct line"]),
    (Category.loan_repayment, ["loan", "repayment", "zopa", "carfinance"]),
    (Category.groceries, ["tesco", "sainsbury", "aldi", "lidl", "asda", "morrison", "co-op"]),
    (Category.transport, ["tfl", "fuel", "shell", "trainline", "stagecoach"]),
    (Category.dining, ["deliveroo", "pret", "nandos", "greggs", "just eat", "restaurant"]),
    (Category.shopping, ["amazon", "asos", "argos", "ebay", "h and m"]),
    (Category.entertainment, ["cinema", "vue", "odeon", "steam", "playstation"]),
]


def rule_category(description: str, amount: float, raw_type: str = "") -> tuple[Category, float]:
    """Return (category, confidence) from the deterministic rules."""
    t = description.lower()
    # own-account transfers, either direction ("TRANSFER ... ACCOUNT")
    if "transfer" in t and "account" in t:
        return Category.internal_transfer, 0.95
    if "transfer to savings" in t:
        return Category.savings, 0.95
    rules = CREDIT_RULES if amount > 0 else DEBIT_RULES
    for cat, kws in rules:
        if any(k in t for k in kws):
            return cat, 0.95
    if amount > 0 and raw_type == "BGC":  # bank-giro credit hint, weaker evidence
        return Category.income, 0.85
    if amount < 0 and raw_type == "FEE":
        return Category.fees, 0.9
    if amount < 0 and raw_type == "ATM":
        return Category.cash_withdrawal, 0.9
    return Category.unknown, 0.30  # abstain rather than guess


def _mk(t: dict, cat: Category, conf: float, source: str) -> CategorizedTransaction:
    return CategorizedTransaction(
        transaction_id=str(t["transaction_id"]),
        account_id=str(t["account_id"]),
        date=t["date"],
        description=t["description"],
        amount=float(t["amount"]),
        raw_type=str(t.get("raw_type", "") or ""),
        category=cat,
        confidence=conf,
        source=source,
        is_essential=cat in ESSENTIAL,
    )


def categorize_rules(txns: list[dict]) -> tuple[list[CategorizedTransaction], dict]:
    out = []
    for t in txns:
        cat, conf = rule_category(t["description"], float(t["amount"]), str(t.get("raw_type", "")))
        out.append(_mk(t, cat, conf, "rules"))
    meta = _meta_base("rules")
    meta["unknown_count"] = sum(1 for c in out if c.category == Category.unknown)
    return out, meta


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------


def _meta_base(mode: str) -> dict:
    return dict(
        mode=mode,
        prompt_version=prompts.PROMPT_VERSION,
        batches=0,
        llm_calls=0,
        parse_failures=0,
        repairs=0,
        repaired_items=0,
        fallback_items=0,
        invented_ids=0,
        sign_rejected=0,
        unknown_count=0,
        llm_unavailable=False,
        stats=CallStats(),
    )


def _validate_batch(
    raw: dict, input_ids: set[str]
) -> tuple[dict[str, tuple[Category, float]], list[str], int]:
    """Pydantic-validate a raw model reply. Returns (id->assignment, missing, invented)."""
    parsed = CategorizationBatchResult.model_validate(raw)
    missing, invented = parsed.validate_against_inputs(input_ids)
    ok = {
        i.transaction_id: (i.category, i.confidence)
        for i in parsed.items
        if i.transaction_id in input_ids
    }
    return ok, sorted(missing), len(invented)


def categorize_llm(txns: list[dict], llm: BaseLLM) -> tuple[list[CategorizedTransaction], dict]:
    cfg = llm_config()
    meta = _meta_base(f"llm:{llm.name}/{llm.model}")
    system = prompts.categorize_system()
    assigned: dict[str, tuple[Category, float, str]] = {}

    for start in range(0, len(txns), cfg.batch_size):
        batch = txns[start : start + cfg.batch_size]
        ids = {str(t["transaction_id"]) for t in batch}
        meta["batches"] += 1
        user = prompts.categorize_user(batch)
        source = "llm"
        try:
            for attempt in (0, 1):  # initial + one repair
                meta["llm_calls"] += 1
                r = llm.chat(system, user, json_mode=True, stats=meta["stats"])
                try:
                    ok, missing, invented = _validate_batch(extract_json(r.text), ids)
                    meta["invented_ids"] += invented
                except (ValueError, ValidationError) as e:
                    meta["parse_failures"] += 1
                    if attempt == 0:
                        meta["repairs"] += 1
                        user = (
                            prompts.categorize_user(batch)
                            + "\n\n"
                            + prompts.categorize_repair(str(e)[:300], [])
                        )
                        source = "llm_repair"
                        continue
                    ok, missing = {}, sorted(ids)  # repair failed too
                    break
                if not missing:
                    break
                if attempt == 0:  # coverage repair
                    meta["repairs"] += 1
                    user = (
                        prompts.categorize_user(batch)
                        + "\n\n"
                        + prompts.categorize_repair("missing items", missing)
                    )
                    source = "llm_repair"
            for tid, (cat, conf) in ok.items():
                assigned[tid] = (cat, conf, source)
                if source == "llm_repair":
                    meta["repaired_items"] += 1
        except LLMUnavailable:
            meta["llm_unavailable"] = True
            break  # rules fallback for the rest

    out = []
    for t in txns:
        tid = str(t["transaction_id"])
        if tid in assigned:
            cat, conf, source = assigned[tid]
            try:
                out.append(_mk(t, cat, conf, source))
                continue
            except ValidationError:  # sign-inconsistent category (e.g. income on a debit)
                meta["sign_rejected"] += 1
        # never an invalid / missing / sign-inconsistent value: deterministic fallback
        cat, conf = rule_category(t["description"], float(t["amount"]), str(t.get("raw_type", "")))
        out.append(_mk(t, cat, conf, "rule_fallback"))
        meta["fallback_items"] += 1
    meta["unknown_count"] = sum(1 for c in out if c.category == Category.unknown)
    return out, meta


def categorize(
    txns: list[dict], llm: BaseLLM | None = None
) -> tuple[list[CategorizedTransaction], dict]:
    if llm is None:
        return categorize_rules(txns)
    return categorize_llm(txns, llm)
