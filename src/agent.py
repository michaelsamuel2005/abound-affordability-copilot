"""Assessment pipeline: categorise -> affordability -> policy retrieval -> rules
-> cited, validated recommendation for human review.

Honest framing of "agentic": orchestration is a FIXED Python pipeline with
state-dependent routing — the LLM does not pick tools. What the LLM does do
(when configured) is bounded and validated: categorise transactions inside a
Pydantic contract, and draft the rationale, which is then checked so that every
cited policy ID exists in the retrieved/decisive set and every number appears in
the computed facts; a rationale that fails validation is REJECTED and replaced
by the deterministic template. The regulated outcome always comes from
`decision_rules.py`. This split — LLM for language, rules for the decision — is
deliberate for a regulated domain.
"""

from __future__ import annotations

import re

from affordability import compute_metrics, max_amount_for_repayment, monthly_repayment
from categorize import categorize
from config import ENGINE_VERSION, POLICY_VERSION, Thresholds, thresholds
from decision_rules import GUARDRAIL_POLICY, evaluate_rules
from llm import BaseLLM, CallStats, LLMUnavailable
from obs import applicant_id_var, get_logger, log_event, stage_timer
from prompts import PROMPT_VERSION, rationale_system, rationale_user
from schemas import LendingDecision, Outcome, PolicyCitation, Transaction, WarningCode, make_warning

log = get_logger("copilot.agent")


# ---------------------------------------------------------------------------
# Retrieval query: built deterministically from the computed metrics, so the
# passages retrieved match the case's actual risk signals (state-dependent).
# ---------------------------------------------------------------------------


def build_query(
    m, cfg: Thresholds, dti_new: float = 0.0, disposable_after: float | None = None
) -> str:
    parts = ["affordability disposable income buffer repayment maximum affordable amount"]
    # query v3: loan-INCLUSIVE signals (v2 only saw pre-loan metrics, so a
    # DTI-driven decline could retrieve nothing about the DTI limit)
    if dti_new > cfg.dti_refer:
        parts.append("debt-to-income including the proposed new loan limit exceeded declined")
    if disposable_after is not None and disposable_after < cfg.buffer_gbp:
        parts.append("minimum disposable income buffer reduce amount maximum affordable")
    if (
        m.n_internal_transfer_txns
        and m.internal_transfer_gross > 0
        and abs(m.internal_transfer_net)
        > max(
            cfg.transfer_imbalance_floor_gbp,
            cfg.transfer_imbalance_tolerance * m.internal_transfer_gross,
        )
    ) or (m.essential_spend == 0 and m.n_transactions >= cfg.min_transactions):
        parts.append("classification plausibility transfer legs balance essential spending")
    if m.months_observed < cfg.min_months or m.n_transactions < cfg.min_transactions:
        parts.append("insufficient data thin file minimum history transactions")
    if m.monthly_income_assessed <= 0 or m.income_months < cfg.min_income_months:
        parts.append("income recognition no eligible income unrecognised inflows")
    if m.coverage_gap:
        parts.append("missing month contiguous coverage gap")
    if m.unknown_share > cfg.max_unknown_share:
        parts.append("unclassified unknown transactions manual review")
    if m.cash_share > cfg.max_cash_share:
        parts.append("cash withdrawals unverifiable spending")
    if m.gambling_ratio > cfg.gambling_refer:
        parts.append("gambling harmful spend")
    if m.distress_events >= cfg.distress_max:
        parts.append("overdraft returned direct debit financial distress")
    if m.income_volatility > cfg.volatility_max:
        parts.append("income volatility conservative estimate verification stability")
    if m.benefits_share > cfg.benefits_share_review:
        parts.append("benefits vulnerability manual review")
    if m.dti_existing > cfg.dti_refer * 0.6:
        parts.append("debt-to-income limit")
    # data treatments actually applied to THIS case (query v2: one phrase per
    # treatment — a single blanket phrase let DQ-002 outrank DQ-003 on refund cases)
    if m.n_internal_transfer_txns:
        parts.append("internal transfers own accounts excluded money moved")
    if m.n_refunds_netted:
        parts.append("refunds reversals netted against original spending never income")
    if m.n_duplicates_removed:
        parts.append("duplicate postings detected removed")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Rationale validation: the LLM's narrative may only cite retrieved/decisive
# policy IDs and may only use numbers that appear in the supplied facts.
# ---------------------------------------------------------------------------

# A leading minus belongs to the number ONLY when it is not preceded by a digit
# (so "£-458" parses as -458, while "12-36 months" still parses as 12 and 36).
# Without the sign, a rationale quoting a negative fact as "£-458" extracted
# "458", which never matched the allowed form "-458" -> a false ungrounded-number
# violation on every applicant with a negative post-repayment buffer. Found by
# the live-LLM run of 2026-08-01; regression-tested in test_agent.py.
_NUM_RE = re.compile(r"(?<!\d)-?\d[\d,]*(?:\.\d+)?")
_ID_RE = re.compile(r"\[((?:POL|DQ)-\d+)\]")


# only these facts are ratios, so only they may also appear in percentage form —
# adding x*100 for EVERY fact would let "£300,000" pass when income is £3,000
RATIO_FACTS = {
    "dti_including_new",
    "gambling_ratio",
    "income_volatility",
    "unknown_share",
    "cash_share",
    "benefits_share",
    "essential_share",
}


def _allowed_numbers(facts: dict, cfg: Thresholds) -> set[str]:
    vals: set[str] = set()

    def add(x: float) -> None:
        for s in (f"{x:.0f}", f"{x:.1f}", f"{x:.2f}", f"{x:,.0f}", f"{x:,.2f}"):
            vals.add(s.replace(",", ""))

    for k, v in facts.items():
        if isinstance(v, int | float):
            add(float(v))
            if k in RATIO_FACTS and 0 <= float(v) <= 1.5:
                add(abs(round(float(v) * 100)))  # ratio -> percentage form
                add(round(float(v) * 100, 1))
    for t in (
        cfg.buffer_gbp,
        cfg.dti_refer * 100,
        cfg.dti_max * 100,
        cfg.gambling_refer * 100,
        cfg.max_unknown_share * 100,
        cfg.max_cash_share * 100,
        cfg.volatility_max,
        cfg.min_months,
        cfg.min_transactions,
        cfg.distress_max,
        cfg.reduced_offer_min_gbp,
        cfg.benefits_share_review * 100,
        cfg.essential_share_high * 100,
    ):
        add(float(t))
    return vals


def validate_rationale(
    text: str, facts: dict, allowed_ids: set[str], cfg: Thresholds
) -> tuple[bool, list[str]]:
    problems = []
    cited = set(_ID_RE.findall(text))
    bad_ids = cited - allowed_ids
    if bad_ids:
        problems.append(f"cites policy IDs outside the retrieved/decisive set: {sorted(bad_ids)}")
    if not cited:
        problems.append("cites no policy ID")
    allowed = _allowed_numbers(facts, cfg)
    for raw in _NUM_RE.findall(_ID_RE.sub(" ", text)):  # ignore digits inside [POL-xxx] ids
        n = raw.replace(",", "")
        forms = {n}
        try:
            forms |= {f"{float(n):.0f}", f"{float(n):.1f}", f"{float(n):.2f}"}
        except ValueError:
            pass
        if not (forms & allowed):
            problems.append(f"contains a number not present in the computed facts: {raw}")
    return (not problems), problems


def _template_rationale(reasons: list[str], pids: list[str]) -> str:
    cite = " ".join(f"[{p}]" for p in pids)
    return f"{' '.join(reasons)} Policy basis: {cite}."


def build_facts(
    m,
    requested: float,
    term: int,
    repayment: float,
    disposable_after: float,
    dti_new: float,
    max_afford: float,
) -> dict:
    """The ONLY numbers the rationale (LLM or template) may use — also the
    reference set the faithfulness checker validates against.

    Every numeric field of the metrics object is included, not a hand-picked
    subset. The subset was a latent defect: deterministic guardrail messages are
    spliced verbatim into the template rationale, so any warning quoting a metric
    outside the subset made the template fail the system's own grounding check.
    The live-LLM run of 2026-08-01 hit exactly that via DQ-007's transfer-imbalance
    message ("net £775 vs gross £775" — neither field was a fact). Enumerating the
    metrics keeps the invariant "every number is a computed fact" true by
    construction. Percentage (x100) forms remain restricted to RATIO_FACTS, so the
    audit-found fact-x100 hole stays closed.
    """
    facts = {
        k: v
        for k, v in m.model_dump().items()
        if isinstance(v, int | float) and not isinstance(v, bool)
    }
    facts.update(
        {
            "requested_amount": requested,
            "term_months": term,
            "monthly_repayment": repayment,
            "disposable_after_repayment": disposable_after,
            "dti_including_new": dti_new,
            "max_affordable_amount": max_afford,
        }
    )
    return facts


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def assess(
    applicant: dict,
    txns: list[dict],
    retriever,
    llm: BaseLLM | None = None,
    cfg: Thresholds | None = None,
) -> tuple[LendingDecision, dict]:
    """Run one full assessment. Returns (decision, diagnostics)."""
    cfg = cfg or thresholds()
    applicant_id = str(applicant["applicant_id"])
    applicant_id_var.set(applicant_id)
    timings: dict[str, float] = {}

    with stage_timer(timings, "validate_input_ms"):
        validated = [Transaction.model_validate(t) for t in txns]

    with stage_timer(timings, "categorize_ms"):
        cats, cat_meta = categorize([t.model_dump() for t in validated], llm=llm)
    log_event(
        log,
        "stage.categorize",
        mode=cat_meta["mode"],
        n=len(cats),
        unknown=cat_meta["unknown_count"],
        repairs=cat_meta["repairs"],
        fallback_items=cat_meta["fallback_items"],
    )

    with stage_timer(timings, "affordability_ms"):
        m = compute_metrics(cats, cfg)

    requested = float(applicant["requested_amount"])
    term = int(applicant["term_months"])
    repayment = monthly_repayment(requested, term, cfg.apr)
    disposable_after = round(m.disposable_income - repayment, 2)
    if m.monthly_income_assessed > 0:
        dti_new = round((m.existing_debt_repayments + repayment) / m.monthly_income_assessed, 3)
        max_afford = max_amount_for_repayment(
            max(m.disposable_income - cfg.buffer_gbp, 0.0), term, cfg.apr
        )
    else:
        dti_new, max_afford = 0.0, 0.0

    with stage_timer(timings, "retrieve_ms"):
        try:
            retrieved, ret_meta = retriever.retrieve(
                build_query(m, cfg, dti_new=dti_new, disposable_after=disposable_after)
            )
        except Exception:  # a retrieval failure must degrade to refer, not 500
            log.exception("retrieval failed; continuing with empty context")
            retrieved, ret_meta = (
                [],
                {
                    "k": 0,
                    "returned": 0,
                    "ids": [],
                    "scores": [],
                    "latency_ms": 0.0,
                    "min_score": 0.0,
                    "empty": True,
                    "error": "retrieval_exception",
                },
            )
    log_event(
        log, "stage.retrieve", **{k: ret_meta[k] for k in ("k", "returned", "ids", "latency_ms")}
    )

    with stage_timer(timings, "decide_ms"):
        rr = evaluate_rules(
            m,
            requested,
            term,
            repayment,
            disposable_after,
            dti_new,
            max_afford,
            cfg,
            categorize_meta=cat_meta,
            retrieval_empty=ret_meta["empty"],
        )

    # deterministic citations: assembled from corpus metadata, never model text
    retrieved_ids = {c["policy_id"] for c in retrieved}
    scores = {c["policy_id"]: c["score"] for c in retrieved}
    citations = []
    for pid in rr.decisive_policy_ids:
        c = retriever.by_id.get(pid)
        if c:
            citations.append(
                PolicyCitation(
                    policy_id=pid,
                    doc_id=c["doc_id"],
                    title=c["title"],
                    version=c["version"],
                    quote=c["body"][:380],
                    score=scores.get(pid),
                )
            )
    retrieval_hits = [pid in retrieved_ids for pid in rr.decisive_policy_ids]

    warnings = list(rr.warnings)
    facts = build_facts(m, requested, term, repayment, disposable_after, dti_new, max_afford)

    rationale = _template_rationale(rr.reasons, rr.decisive_policy_ids)
    rationale_source = "template"
    if llm is not None:
        with stage_timer(timings, "rationale_ms"):
            try:
                stats = CallStats()
                passages = retrieved + [
                    retriever.by_id[p]
                    for p in rr.decisive_policy_ids
                    if p not in retrieved_ids and p in retriever.by_id
                ]
                r = llm.chat(
                    rationale_system(),
                    rationale_user(facts, passages, rr.outcome.value),
                    stats=stats,
                )
                ok, problems = validate_rationale(
                    r.text, facts, retrieved_ids | set(rr.decisive_policy_ids), cfg
                )
                if ok:
                    rationale, rationale_source = r.text.strip(), "llm"
                else:
                    warnings.append(
                        make_warning(
                            WarningCode.rationale_rejected,
                            "LLM rationale failed grounding validation and was replaced "
                            f"by the deterministic template: {'; '.join(problems[:3])}",
                        )
                    )
                    log_event(log, "rationale.rejected", problems=problems[:5])
            except LLMUnavailable:
                warnings.append(
                    make_warning(
                        WarningCode.llm_unavailable,
                        "LLM provider unavailable for rationale; deterministic template used.",
                    )
                )

    guard = rr.guardrail or next((w.code for w in warnings if w.severity == "guardrail"), None)
    outcome = Outcome.refer if guard else rr.outcome  # late guardrails also force refer

    confidence, reasons = rr.confidence, list(rr.reasons)
    decisive_ids = list(rr.decisive_policy_ids)
    if outcome != rr.outcome:
        # a guardrail raised AFTER the rules ran (e.g. rationale-stage provider
        # outage) flipped the outcome — the rest of the decision must match it,
        # not describe the pre-flip result
        late = next(w for w in warnings if w.severity == "guardrail" and w.code == guard)
        confidence = 0.5
        reasons = [
            f"Referred for manual review ({late.code.value}): {late.message}",
            f"Pre-referral rules assessment for context: {rr.outcome.value} — "
            + " ".join(rr.reasons),
        ]
        guard_pid = GUARDRAIL_POLICY.get(guard, "POL-010")
        if guard_pid not in decisive_ids:
            decisive_ids.insert(0, guard_pid)
            gchunk = retriever.by_id.get(guard_pid)
            if gchunk:
                citations.insert(
                    0,
                    PolicyCitation(
                        policy_id=guard_pid,
                        doc_id=gchunk["doc_id"],
                        title=gchunk["title"],
                        version=gchunk["version"],
                        quote=gchunk["body"][:380],
                        score=scores.get(guard_pid),
                    ),
                )
        rationale = _template_rationale(reasons, decisive_ids)

    # citation completeness: any policy ID the accepted rationale cites must have
    # a structured citation object (retrieved-only IDs included)
    for pid in _ID_RE.findall(rationale):
        if pid in retriever.by_id and all(c.policy_id != pid for c in citations):
            c = retriever.by_id[pid]
            citations.append(
                PolicyCitation(
                    policy_id=pid,
                    doc_id=c["doc_id"],
                    title=c["title"],
                    version=c["version"],
                    quote=c["body"][:380],
                    score=scores.get(pid),
                )
            )

    decision = LendingDecision(
        applicant_id=applicant_id,
        outcome=outcome,
        confidence=confidence,
        requested_amount=requested,
        term_months=term,
        monthly_repayment=repayment,
        max_affordable_amount=max_afford,
        disposable_after_repayment=disposable_after,
        dti_including_new=dti_new,
        reasons=reasons,
        rationale=rationale,
        warnings=warnings,
        guardrail=guard,
        policy_citations=citations,
        human_review_required=True,
        review_priority="high" if (guard or outcome != Outcome.approve) else "standard",
        versions={
            "engine": ENGINE_VERSION,
            "policy_corpus": POLICY_VERSION,
            "prompts": PROMPT_VERSION,
            "llm": f"{llm.name}/{llm.model}" if llm else "deterministic",
            "embeddings": retriever.embedder.name,
        },
    )

    diag = {
        "metrics": m.model_dump(),
        "categorize_meta": {k: v for k, v in cat_meta.items() if k != "stats"},
        "categorize_tokens": {
            "prompt": cat_meta["stats"].prompt_tokens,
            "completion": cat_meta["stats"].completion_tokens,
        },
        "retrieval": ret_meta,
        "retrieved_ids": sorted(retrieved_ids),
        "decisive_ids": rr.decisive_policy_ids,
        "retrieval_hits": retrieval_hits,
        "rationale_source": rationale_source,
        "timings_ms": timings,
        "categories": {c.transaction_id: c.category.value for c in cats},
    }
    log_event(
        log,
        "decision",
        outcome=outcome.value,
        guardrail=str(guard),
        dti=dti_new,
        disposable_after=disposable_after,
        warnings=[w.code.value for w in warnings],
        cited=[c.policy_id for c in citations],
        timings_ms=timings,
        llm=decision.versions["llm"],
        prompt_version=PROMPT_VERSION,
    )
    return decision, diag
