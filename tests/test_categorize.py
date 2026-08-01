"""Categoriser tests: deterministic rules (especially the inflow asymmetry — an
unrecognised credit is never income) and the LLM validate-and-repair loop with
scripted model behaviour: malformed JSON, invalid enums, invented/missing IDs,
provider outage."""

import json

from fakes import BrokenLLM, FakeLLM, valid_items_json

from categorize import categorize, categorize_rules, rule_category
from schemas import Category

# ---------------------------------------------------------------------------
# rules path
# ---------------------------------------------------------------------------

CASES = [
    ("ACME LTD SALARY", 2400, "BGC", Category.income),
    ("UBER PARTNER PAYMENT", 310, "FPI", Category.income),
    ("DWP UC PAYMENT", 900, "FPI", Category.benefits),
    ("AMAZON.CO.UK REFUND", 61.5, "FPI", Category.refund),
    ("TRANSFER FROM CURRENT ACCOUNT", 250, "TFR", Category.internal_transfer),
    ("TRANSFER TO SAVINGS ACCOUNT", -250, "TFR", Category.internal_transfer),
    ("TRANSFER TO SAVINGS", -300, "TFR", Category.savings),
    ("MONZO POT", -100, "TFR", Category.savings),
    ("PAYPAL TRANSFER J SMITH", 1500, "FPI", Category.unknown),  # credit != income
    ("RENT PROPERTYCO", -900, "DD", Category.rent_mortgage),
    ("OCTOPUS ENERGY", -80, "DD", Category.utilities),
    ("LB COUNCIL TAX", -140, "DD", Category.council_tax),
    ("KLARNA", -45, "DEB", Category.bnpl),
    ("BET365", -20, "DEB", Category.gambling),
    ("CASH WDL LLOYDS ATM", -100, "ATM", Category.cash_withdrawal),
    ("OVERDRAFT FEE", -10, "FEE", Category.fees),
    ("AMAZON PRIME", -8.99, "DEB", Category.subscriptions),  # 'prime' beats 'amazon'
    ("AMAZON.CO.UK", -30, "DEB", Category.shopping),
    ("CARFINANCE 247 LOAN", -210, "DD", Category.loan_repayment),
    ("SQ *BLUE TIT COFFEE", -4.5, "DEB", Category.unknown),  # ambiguous -> abstain
]


def test_rule_categories():
    for desc, amount, rt, expected in CASES:
        got, conf = rule_category(desc, amount, rt)
        assert got == expected, f"{desc}: {got} != {expected}"
        assert 0 < conf <= 1


def test_unknown_inflow_is_never_income():
    got, conf = rule_category("RANDOM CREDIT 12345", 5000, "FPI")
    assert got == Category.unknown and conf < 0.5


def test_rules_meta_counts_unknowns(mk_txn):
    txns = [
        mk_txn(description="SQ *MYSTERY", amount=-9.0),
        mk_txn(description="TESCO STORES", amount=-20.0),
    ]
    cats, meta = categorize_rules(txns)
    assert meta["unknown_count"] == 1 and meta["mode"] == "rules"
    assert cats[1].is_essential is True


# ---------------------------------------------------------------------------
# LLM path: validate-and-repair
# ---------------------------------------------------------------------------


def test_llm_valid_first_pass(mk_txn):
    txns = [mk_txn(), mk_txn(description="NANDOS", amount=-18.0)]
    llm = FakeLLM(
        [
            valid_items_json(
                txns,
                overrides={
                    txns[0]["transaction_id"]: "groceries",
                    txns[1]["transaction_id"]: "dining",
                },
            )
        ]
    )
    cats, meta = categorize(txns, llm=llm)
    assert [c.category for c in cats] == [Category.groceries, Category.dining]
    assert all(c.source == "llm" for c in cats)
    assert meta["repairs"] == 0 and meta["fallback_items"] == 0
    assert llm.calls[0]["json_mode"] is True
    assert "<transactions>" in llm.calls[0]["user"]  # injection delimiters


def test_llm_malformed_then_repaired(mk_txn):
    txns = [mk_txn()]
    llm = FakeLLM(["not json at all {{", valid_items_json(txns)])
    cats, meta = categorize(txns, llm=llm)
    assert cats[0].category == Category.groceries
    assert cats[0].source == "llm_repair"
    assert meta["parse_failures"] == 1 and meta["repairs"] == 1
    assert "failed validation" in llm.calls[1]["user"]


def test_llm_malformed_twice_falls_back_to_rules(mk_txn):
    txns = [mk_txn(description="BET365", amount=-50.0)]
    llm = FakeLLM(["garbage", "still garbage"])
    cats, meta = categorize(txns, llm=llm)
    assert cats[0].category == Category.gambling  # rules got it right
    assert cats[0].source == "rule_fallback"
    assert meta["fallback_items"] == 1 and meta["parse_failures"] == 2


def test_llm_invalid_category_triggers_repair(mk_txn):
    txns = [mk_txn()]
    bad = json.dumps(
        {
            "items": [
                {
                    "transaction_id": txns[0]["transaction_id"],
                    "category": "crypto_stuff",
                    "confidence": 0.9,
                }
            ]
        }
    )
    llm = FakeLLM([bad, valid_items_json(txns)])
    cats, meta = categorize(txns, llm=llm)
    assert cats[0].category == Category.groceries and meta["repairs"] == 1


def test_llm_invented_ids_dropped_and_counted(mk_txn):
    txns = [mk_txn()]
    reply = json.dumps(
        {
            "items": [
                {
                    "transaction_id": txns[0]["transaction_id"],
                    "category": "groceries",
                    "confidence": 0.9,
                },
                {"transaction_id": "TX-INVENTED", "category": "income", "confidence": 0.99},
            ]
        }
    )
    cats, meta = categorize(txns, llm=FakeLLM([reply]))
    assert len(cats) == 1 and meta["invented_ids"] == 1
    assert cats[0].category == Category.groceries


def test_llm_missing_ids_repaired_then_fallback(mk_txn):
    txns = [mk_txn(), mk_txn(description="RENT PROPERTYCO", amount=-900.0, raw_type="DD")]
    only_first = json.dumps(
        {
            "items": [
                {
                    "transaction_id": txns[0]["transaction_id"],
                    "category": "groceries",
                    "confidence": 0.9,
                }
            ]
        }
    )
    llm = FakeLLM([only_first, only_first])  # never covers the second txn
    cats, meta = categorize(txns, llm=llm)
    assert cats[1].category == Category.rent_mortgage
    assert cats[1].source == "rule_fallback"
    assert meta["repairs"] == 1 and meta["fallback_items"] == 1


def test_provider_outage_falls_back_entirely(mk_txn):
    txns = [mk_txn(), mk_txn(description="SPOTIFY", amount=-9.99)]
    cats, meta = categorize(txns, llm=BrokenLLM())
    assert meta["llm_unavailable"] is True
    assert all(c.source == "rule_fallback" for c in cats)
    assert cats[1].category == Category.subscriptions


def test_every_input_has_exactly_one_output(mk_txn):
    txns = [mk_txn() for _ in range(7)]
    cats, _ = categorize(txns, llm=FakeLLM([valid_items_json(txns[:3])]))
    assert len(cats) == len(txns)
    assert {c.transaction_id for c in cats} == {t["transaction_id"] for t in txns}


def test_is_essential_comes_from_our_mapping_not_the_model(mk_txn):
    txns = [mk_txn(description="BET365", amount=-50.0)]
    cats, _ = categorize(txns, llm=FakeLLM([valid_items_json(txns, category="gambling")]))
    assert cats[0].is_essential is False


def test_llm_sign_inconsistent_category_falls_back_to_rules(mk_txn):
    """A schema-valid-looking but sign-inconsistent assignment (income on a
    debit) must be rejected and re-categorised deterministically."""
    txns = [mk_txn(description="RENT PROPERTYCO", amount=-900.0, raw_type="DD")]
    hostile = valid_items_json(txns, category="income", confidence=0.99)
    cats, meta = categorize(txns, llm=FakeLLM([hostile, hostile]))
    assert cats[0].category == Category.rent_mortgage
    assert cats[0].source == "rule_fallback"
    assert meta["sign_rejected"] >= 1 and meta["fallback_items"] == 1
