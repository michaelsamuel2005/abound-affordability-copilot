"""FastAPI integration tests via TestClient: the full request/response cycle,
validation failures, size limits, idempotency, and the functional human-review
loop (uphold, override, category corrections -> deterministic recompute)."""

import pytest
from conftest import make_txn
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app import app

    with TestClient(app) as c:  # context manager runs the lifespan
        yield c


def _healthy_payload(months=6, applicant="API-001", **over):
    txns = []
    for mm in range(1, months + 1):
        m = f"2026-{mm:02d}"
        txns.append(
            make_txn(description="ACME LTD SALARY", amount=3000.0, date=f"{m}-28", raw_type="BGC")
        )
        txns.append(
            make_txn(description="RENT PROPERTYCO", amount=-900.0, date=f"{m}-01", raw_type="DD")
        )
        for wk in range(4):
            txns.append(
                make_txn(description="TESCO STORES", amount=-80.0, date=f"{m}-{3 + wk * 7:02d}")
            )
        for i in range(5):
            txns.append(make_txn(description="PRET A MANGER", amount=-9.5, date=f"{m}-{5 + i:02d}"))
    payload = {
        "applicant_id": applicant,
        "requested_amount": 4000.0,
        "term_months": 24,
        "transactions": txns,
    }
    payload.update(over)
    return payload


def test_health_ready_version(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready").json()
    assert r["status"] == "ready" and r["chunks"] == 17
    v = client.get("/version").json()
    assert v["prompts"] == "v3" and v["llm"] == "deterministic"
    assert v["thresholds"]["dti_max"] == 0.45


def test_decision_endpoint_full_cycle(client):
    r = client.post("/v1/decision", json=_healthy_payload())
    assert r.status_code == 201
    body = r.json()
    d = body["decision"]
    assert d["outcome"] == "approve" and d["human_review_required"] is True
    assert body["assessment_id"].startswith("ASMT-")
    assert body["metrics"]["monthly_income_assessed"] == 3000.0
    assert d["policy_citations"] and d["versions"]["engine"]
    assert "X-Request-ID" in r.headers

    got = client.get(f"/v1/assessments/{body['assessment_id']}")
    assert got.status_code == 200
    assert got.json()["decision"]["outcome"] == "approve"
    with_tx = client.get(
        f"/v1/assessments/{body['assessment_id']}", params={"include_transactions": True}
    ).json()
    assert with_tx["transactions"] and with_tx["categories"]


def test_idempotency_replay(client):
    p = _healthy_payload(applicant="API-IDEM", idempotency_key="idem-123")
    first = client.post("/v1/decision", json=p)
    second = client.post("/v1/decision", json=p)
    assert first.status_code == 201 and second.status_code == 200
    assert second.headers.get("X-Idempotent-Replay") == "true"
    assert first.json()["assessment_id"] == second.json()["assessment_id"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["transactions"].append(dict(p["transactions"][0])),  # dup id
        lambda p: p["transactions"][0].update(amount=0),  # zero amount
        lambda p: p["transactions"][0].update(date="01-04-2026"),  # bad date
        lambda p: p["transactions"][0].update(extra_field=1),  # invented field
        lambda p: p.update(requested_amount=-5),
        lambda p: p.update(term_months=3),  # < 6 months
    ],
)
def test_validation_errors_return_422(client, mutate):
    p = _healthy_payload(applicant="API-BAD")
    mutate(p)
    assert client.post("/v1/decision", json=p).status_code == 422


def test_oversized_request_rejected_413(client, monkeypatch):
    monkeypatch.setenv("MAX_TRANSACTIONS", "10")
    r = client.post("/v1/decision", json=_healthy_payload(applicant="API-BIG"))
    assert r.status_code == 413


def test_unknown_assessment_404(client):
    assert client.get("/v1/assessments/ASMT-nope").status_code == 404
    r = client.post(
        "/v1/assessments/ASMT-nope/review",
        json={"reviewer": "r", "action": "uphold", "reason": "does not matter here"},
    )
    assert r.status_code == 404


def test_review_uphold_and_audit_trail(client):
    aid = client.post("/v1/decision", json=_healthy_payload(applicant="API-REV1")).json()[
        "assessment_id"
    ]
    r = client.post(
        f"/v1/assessments/{aid}/review",
        json={
            "reviewer": "underwriter-1",
            "action": "uphold",
            "reason": "checked the evidence, agree with approve",
        },
    )
    assert r.status_code == 201
    rec = r.json()
    assert rec["original_outcome"] == rec["final_outcome"] == "approve"
    stored = client.get(f"/v1/assessments/{aid}").json()
    assert len(stored["reviews"]) == 1
    assert stored["reviews"][0]["reviewer"] == "underwriter-1"


def test_review_override_requires_outcome_and_reason(client):
    aid = client.post("/v1/decision", json=_healthy_payload(applicant="API-REV2")).json()[
        "assessment_id"
    ]
    r = client.post(
        f"/v1/assessments/{aid}/review",
        json={
            "reviewer": "u",
            "action": "override",
            "reason": "override without outcome should fail",
        },
    )
    assert r.status_code == 422
    r = client.post(
        f"/v1/assessments/{aid}/review",
        json={
            "reviewer": "u",
            "action": "override",
            "final_outcome": "refer",
            "reason": "escalating for a second opinion",
        },
    )
    assert r.status_code == 201 and r.json()["final_outcome"] == "refer"


def test_review_category_corrections_trigger_recompute(client):
    payload = _healthy_payload(applicant="API-REV3")
    created = client.post("/v1/decision", json=payload).json()
    aid = created["assessment_id"]
    assert created["decision"]["outcome"] == "approve"

    # reviewer discovers the "groceries" were actually gambling (24 txns x £80
    # = £320/month vs £3,000 income -> ratio 10.7% > 10%) -> recompute refers
    grocery_ids = [
        t["transaction_id"] for t in payload["transactions"] if t["description"] == "TESCO STORES"
    ]
    r = client.post(
        f"/v1/assessments/{aid}/review",
        json={
            "reviewer": "underwriter-2",
            "action": "override",
            "final_outcome": "refer",
            "reason": "merchant look-up shows these are gaming merchants",
            "category_corrections": [
                {"transaction_id": t, "category": "gambling"} for t in grocery_ids
            ],
        },
    )
    assert r.status_code == 201
    rec = r.json()
    assert rec["recomputed"] is True
    assert rec["recomputed_outcome"] == "refer"
    assert rec["recomputed_metrics"]["gambling_spend"] > 0


def test_review_corrections_unknown_id_rejected(client):
    aid = client.post("/v1/decision", json=_healthy_payload(applicant="API-REV4")).json()[
        "assessment_id"
    ]
    r = client.post(
        f"/v1/assessments/{aid}/review",
        json={
            "reviewer": "u",
            "action": "uphold",
            "reason": "testing bad correction id",
            "category_corrections": [
                {"transaction_id": "TX-DOES-NOT-EXIST", "category": "gambling"}
            ],
        },
    )
    assert r.status_code == 422


def test_assessment_listing(client):
    listing = client.get("/v1/assessments").json()
    assert any(x["applicant_id"] == "API-001" for x in listing)
    assert all("review_priority" in x for x in listing)


def test_review_sign_inconsistent_correction_rejected(client):
    payload = _healthy_payload(applicant="API-REV5")
    created = client.post("/v1/decision", json=payload).json()
    rent_id = next(
        t["transaction_id"]
        for t in payload["transactions"]
        if t["description"] == "RENT PROPERTYCO"
    )
    r = client.post(
        f"/v1/assessments/{created['assessment_id']}/review",
        json={
            "reviewer": "u",
            "action": "uphold",
            "reason": "testing sign-inconsistent correction",
            "category_corrections": [{"transaction_id": rent_id, "category": "income"}],
        },
    )
    assert r.status_code == 422
    assert "inconsistent with the transaction sign" in r.json()["detail"]


def test_review_audit_trail_persists_corrections_and_metrics(client):
    payload = _healthy_payload(applicant="API-REV6")
    created = client.post("/v1/decision", json=payload).json()
    aid = created["assessment_id"]
    grocery_ids = [
        t["transaction_id"] for t in payload["transactions"] if t["description"] == "TESCO STORES"
    ]
    client.post(
        f"/v1/assessments/{aid}/review",
        json={
            "reviewer": "underwriter-3",
            "action": "override",
            "final_outcome": "refer",
            "reason": "gaming merchants found on manual look-up",
            "category_corrections": [
                {"transaction_id": t, "category": "gambling"} for t in grocery_ids
            ],
        },
    )
    stored = client.get(f"/v1/assessments/{aid}").json()
    rec = stored["reviews"][0]
    assert len(rec["corrections"]) == len(grocery_ids)  # exact edits persisted
    assert rec["corrections"][0]["category"] == "gambling"
    assert rec["recomputed_metrics"]["gambling_spend"] > 0  # post-correction metrics persisted
    assert rec["recomputed_outcome"] == "refer"


def test_review_makes_state_canonical_and_chains(client):
    """Reviews update the CURRENT assessment state: corrected categories become
    the baseline for the next review, and current_outcome reflects the final."""
    payload = _healthy_payload(applicant="API-REV7")
    created = client.post("/v1/decision", json=payload).json()
    aid = created["assessment_id"]
    assert created["current_outcome"] == "approve"

    grocery_ids = [
        t["transaction_id"] for t in payload["transactions"] if t["description"] == "TESCO STORES"
    ]
    first = client.post(
        f"/v1/assessments/{aid}/review",
        json={
            "reviewer": "u1",
            "action": "override",
            "final_outcome": "refer",
            "reason": "half of these groceries are gaming merchants",
            "category_corrections": [
                {"transaction_id": t, "category": "gambling"} for t in grocery_ids[:12]
            ],
        },
    ).json()
    stored = client.get(f"/v1/assessments/{aid}", params={"include_transactions": True}).json()
    assert stored["current_outcome"] == "refer"
    assert stored["current_metrics"]["gambling_spend"] > 0
    assert stored["categories"][grocery_ids[0]] == "gambling"  # canonical now
    assert stored["categories_original"][grocery_ids[0]] == "groceries"  # history kept

    # chained review: starts from the corrected state, so gambling accumulates
    second = client.post(
        f"/v1/assessments/{aid}/review",
        json={
            "reviewer": "u2",
            "action": "override",
            "final_outcome": "refer",
            "reason": "remaining grocery lines are also gaming merchants",
            "category_corrections": [
                {"transaction_id": t, "category": "gambling"} for t in grocery_ids[12:]
            ],
        },
    ).json()
    assert (
        second["recomputed_metrics"]["gambling_spend"]
        > first["recomputed_metrics"]["gambling_spend"]
    )


def test_uphold_conflicting_with_recompute_rejected(client):
    """'Uphold' plus corrections that change the recommendation is incoherent —
    the API forces an explicit override instead of silently keeping either."""
    payload = _healthy_payload(applicant="API-REV8")
    aid = client.post("/v1/decision", json=payload).json()["assessment_id"]
    grocery_ids = [
        t["transaction_id"] for t in payload["transactions"] if t["description"] == "TESCO STORES"
    ]
    r = client.post(
        f"/v1/assessments/{aid}/review",
        json={
            "reviewer": "u",
            "action": "uphold",
            "reason": "relabelling everything as gambling but keeping approve",
            "category_corrections": [
                {"transaction_id": t, "category": "gambling"} for t in grocery_ids
            ],
        },
    )
    assert r.status_code == 422
    assert "use action='override'" in r.json()["detail"]
