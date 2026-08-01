"""FastAPI service: affordability assessment + functional human-review loop.

    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints (API version prefix /v1):
    GET  /health                          liveness (always 200 once process is up)
    GET  /ready                           readiness (503 until index + provider loaded)
    GET  /version                         engine/prompt/policy versions + thresholds
    POST /v1/decision                     run an assessment (idempotency_key supported)
    GET  /v1/assessments                  list stored assessment summaries
    GET  /v1/assessments/{id}             full stored assessment
    POST /v1/assessments/{id}/review      reviewer uphold/override + category
                                          corrections -> deterministic recompute

Design notes:
  * the retriever and LLM client are created ONCE at startup (lifespan), not per
    request; the FAISS index lives in process memory;
  * endpoints are sync `def` — FastAPI runs them on its threadpool; the
    deterministic path is pure CPU and the LLM path uses a blocking HTTP client,
    so a sync worker per request is the simplest correct model at prototype scale;
  * storage is IN-MEMORY (a dict behind a lock): assessments and review records
    do not survive a restart. A real deployment would use Postgres — see
    docs/DECISIONS.md;
  * no authentication/authorisation — documented production gap, not a claim.
"""

from __future__ import annotations

import sys
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from affordability import (  # noqa: E402
    compute_metrics,  # noqa: E402
    max_amount_for_repayment,
    monthly_repayment,
)
from agent import assess  # noqa: E402
from config import ENGINE_VERSION, POLICY_VERSION, api_config, thresholds  # noqa: E402
from decision_rules import evaluate_rules  # noqa: E402
from llm import get_llm  # noqa: E402
from obs import get_logger, log_event, request_id_var  # noqa: E402
from prompts import PROMPT_VERSION  # noqa: E402
from retriever import PolicyRetriever  # noqa: E402
from schemas import (  # noqa: E402
    ESSENTIAL,
    CategorizedTransaction,
    Outcome,
    ReviewActionType,
    ReviewRecord,
    ReviewRequest,
    Transaction,
)

log = get_logger("copilot.api")


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicant_id: str = Field(min_length=1, max_length=64)
    requested_amount: float = Field(gt=0, le=50_000)
    term_months: int = Field(ge=6, le=84)
    transactions: list[Transaction] = Field(min_length=1)
    idempotency_key: str | None = Field(default=None, max_length=128)

    @field_validator("transactions")
    @classmethod
    def _unique_tx_ids(cls, v: list[Transaction]) -> list[Transaction]:
        ids = [t.transaction_id for t in v]
        if len(ids) != len(set(ids)):
            raise ValueError("transaction_id values must be unique within a request")
        return v


class AssessmentSummary(BaseModel):
    assessment_id: str
    applicant_id: str
    outcome: Outcome  # original recommendation
    current_outcome: Outcome  # canonical state after any reviews
    guardrail: str | None
    review_priority: str
    reviewed: bool
    created_at: str


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


class Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.assessments: dict[str, dict] = {}
        self.reviews: dict[str, list[ReviewRecord]] = {}
        self.idempotency: dict[str, str] = {}

    def put(self, record: dict, idem_key: str | None) -> str:
        with self._lock:
            aid = f"ASMT-{len(self.assessments) + 1:05d}-{uuid.uuid4().hex[:6]}"
            record["assessment_id"] = aid
            self.assessments[aid] = record
            if idem_key:
                self.idempotency[idem_key] = aid
            return aid

    def get(self, aid: str) -> dict | None:
        return self.assessments.get(aid)

    def by_idempotency(self, key: str) -> dict | None:
        aid = self.idempotency.get(key)
        return self.assessments.get(aid) if aid else None

    def add_review(self, aid: str, rec: ReviewRecord) -> None:
        with self._lock:
            self.reviews.setdefault(aid, []).append(rec)
            self.assessments[aid]["reviews"].append(rec.model_dump(mode="json"))


state: dict = {"ready": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = PolicyRetriever()
    state["llm"] = get_llm()
    state["store"] = Store()
    state["ready"] = True
    log_event(
        log,
        "startup",
        embeddings=state["retriever"].embedder.name,
        llm=state["llm"].name if state["llm"] else "deterministic",
        chunks=len(state["retriever"].chunks),
    )
    yield
    state["ready"] = False


app = FastAPI(
    title="Abound-style Affordability Copilot",
    version=ENGINE_VERSION,
    description="Explainable cashflow-underwriting recommendations from synthetic "
    "Open-Banking data. Every response is a recommendation for human "
    "review — this service never makes a lending decision.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
    request_id_var.set(rid)
    t0 = perf_counter()
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    log_event(
        log,
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round((perf_counter() - t0) * 1000, 2),
    )
    return response


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Health / meta
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if not state.get("ready"):
        raise HTTPException(503, "index/provider not loaded yet")
    r = state["retriever"]
    return {
        "status": "ready",
        "embeddings": r.embedder.name,
        "chunks": len(r.chunks),
        "llm": state["llm"].name if state["llm"] else "deterministic",
    }


@app.get("/version")
def version():
    cfg = thresholds()
    return {
        "engine": ENGINE_VERSION,
        "policy_corpus": POLICY_VERSION,
        "prompts": PROMPT_VERSION,
        "llm": (
            f"{state['llm'].name}/{state['llm'].model}" if state.get("llm") else "deterministic"
        ),
        "embeddings": state["retriever"].embedder.name if state.get("retriever") else None,
        "thresholds": cfg.__dict__,
    }


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


@app.post("/v1/decision", status_code=201)
def decision(req: DecisionRequest, response: Response):
    if not state.get("ready"):
        raise HTTPException(503, "service not ready")
    if len(req.transactions) > api_config().max_transactions:
        raise HTTPException(413, f"too many transactions (max {api_config().max_transactions})")
    if req.idempotency_key:
        existing = state["store"].by_idempotency(req.idempotency_key)
        if existing:
            response.headers["X-Idempotent-Replay"] = "true"
            response.status_code = 200
            return _public(existing)

    applicant = {
        "applicant_id": req.applicant_id,
        "requested_amount": req.requested_amount,
        "term_months": req.term_months,
    }
    txns = [t.model_dump() for t in req.transactions]
    dec, diag = assess(applicant, txns, state["retriever"], llm=state["llm"])

    record = {
        "applicant_id": req.applicant_id,
        "request": {"requested_amount": req.requested_amount, "term_months": req.term_months},
        "transactions": txns,
        "categories": diag["categories"],  # CURRENT categories (reviews update this)
        "categories_original": dict(diag["categories"]),
        "decision": dec.model_dump(mode="json"),  # original recommendation (immutable)
        "current_outcome": dec.outcome.value,  # canonical state (reviews update this)
        "current_metrics": None,  # set after a recompute
        "metrics": diag["metrics"],
        "diagnostics": {
            k: diag[k]
            for k in (
                "retrieved_ids",
                "decisive_ids",
                "retrieval_hits",
                "rationale_source",
                "timings_ms",
                "categorize_meta",
            )
        },
        "reviews": [],
        "created_at": _now(),
    }
    record["assessment_id"] = state["store"].put(record, req.idempotency_key)
    return _public(record)


def _public(record: dict) -> dict:
    """Response shape: original recommendation + canonical current state."""
    return {
        k: record[k]
        for k in (
            "assessment_id",
            "applicant_id",
            "decision",
            "current_outcome",
            "current_metrics",
            "metrics",
            "diagnostics",
            "reviews",
            "created_at",
        )
    }


@app.get("/v1/assessments")
def list_assessments() -> list[AssessmentSummary]:
    out = []
    for aid, r in state["store"].assessments.items():
        out.append(
            AssessmentSummary(
                assessment_id=aid,
                applicant_id=r["applicant_id"],
                outcome=r["decision"]["outcome"],
                current_outcome=r["current_outcome"],
                guardrail=r["decision"]["guardrail"],
                review_priority=r["decision"]["review_priority"],
                reviewed=bool(r["reviews"]),
                created_at=r["created_at"],
            )
        )
    return out


@app.get("/v1/assessments/{aid}")
def get_assessment(aid: str, include_transactions: bool = False):
    r = state["store"].get(aid)
    if not r:
        raise HTTPException(404, f"assessment {aid} not found")
    out = _public(r)
    if include_transactions:  # reviewer drill-down: raw txns + category states
        out["transactions"] = r["transactions"]
        out["categories"] = r["categories"]  # current
        out["categories_original"] = r["categories_original"]
    return out


# ---------------------------------------------------------------------------
# Human review — uphold / override, with optional category corrections that
# trigger a deterministic recompute (LLM never involved in this path).
# ---------------------------------------------------------------------------


@app.post("/v1/assessments/{aid}/review", status_code=201)
def review(aid: str, req: ReviewRequest):
    r = state["store"].get(aid)
    if not r:
        raise HTTPException(404, f"assessment {aid} not found")

    known = {t["transaction_id"] for t in r["transactions"]}
    bad = [c.transaction_id for c in req.category_corrections if c.transaction_id not in known]
    if bad:
        raise HTTPException(422, f"corrections reference unknown transaction_ids: {bad[:5]}")

    recomputed, recomputed_outcome, recomputed_metrics = False, None, None
    if req.category_corrections:
        corrections = {c.transaction_id: c.category for c in req.category_corrections}
        cats = []
        for t in r["transactions"]:
            tid = t["transaction_id"]
            corrected = tid in corrections
            # chained reviews: start from the CURRENT category state, not the original
            cat = corrections.get(tid) or r["categories"][tid]
            try:
                cats.append(
                    CategorizedTransaction(
                        **t,
                        category=cat,
                        confidence=1.0 if corrected else 0.9,
                        source="human" if corrected else "rules",
                        is_essential=cat in ESSENTIAL,
                    )
                )
            except ValidationError as e:  # e.g. a debit corrected to a credit-only category
                raise HTTPException(
                    422,
                    f"correction for {tid} is inconsistent with the transaction sign: "
                    f"{e.errors()[0].get('msg', 'invalid category')}",
                ) from e
        cfg = thresholds()
        m = compute_metrics(cats, cfg)
        requested = r["request"]["requested_amount"]
        term = r["request"]["term_months"]
        rep = monthly_repayment(requested, term, cfg.apr)
        disp_after = round(m.disposable_income - rep, 2)
        if m.monthly_income_assessed > 0:
            dti = round((m.existing_debt_repayments + rep) / m.monthly_income_assessed, 3)
            max_aff = max_amount_for_repayment(
                max(m.disposable_income - cfg.buffer_gbp, 0), term, cfg.apr
            )
        else:
            dti, max_aff = 0.0, 0.0
        rr = evaluate_rules(m, requested, term, rep, disp_after, dti, max_aff, cfg)
        recomputed, recomputed_outcome, recomputed_metrics = True, rr.outcome, m.model_dump()

    original = Outcome(r["decision"]["outcome"])
    current = Outcome(r["current_outcome"])
    if req.action == ReviewActionType.override:
        final = req.final_outcome
    elif recomputed and recomputed_outcome != current:
        # 'uphold' would be incoherent: the reviewer's own corrections change the
        # recommendation — force an explicit decision instead of a silent one
        raise HTTPException(
            422,
            f"corrections recompute the recommendation to '{recomputed_outcome.value}' "
            f"(currently '{current.value}'); use action='override' with a final_outcome",
        )
    else:
        final = current
    rec = ReviewRecord(
        review_id=f"REV-{uuid.uuid4().hex[:8]}",
        assessment_id=aid,
        reviewer=req.reviewer,
        action=req.action,
        original_outcome=original,
        final_outcome=final,
        reason=req.reason,
        n_corrections=len(req.category_corrections),
        corrections=req.category_corrections,
        recomputed=recomputed,
        recomputed_outcome=recomputed_outcome,
        recomputed_metrics=recomputed_metrics,
        created_at=_now(),
    )
    state["store"].add_review(aid, rec)
    # the reviewed assessment becomes canonical: corrections update the current
    # category map and metrics; the final outcome becomes the current outcome
    with state["store"]._lock:
        if req.category_corrections:
            for c in req.category_corrections:
                r["categories"][c.transaction_id] = c.category.value
            r["current_metrics"] = recomputed_metrics
        r["current_outcome"] = final.value
    log_event(
        log,
        "review",
        assessment_id=aid,
        action=req.action.value,
        original=original.value,
        final=final.value,
        corrections=len(req.category_corrections),
        recomputed_outcome=recomputed_outcome.value if recomputed_outcome else None,
    )

    return rec.model_dump(mode="json")
