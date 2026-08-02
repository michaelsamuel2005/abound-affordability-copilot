"""Evaluation harness — runs the full pipeline against the HELD-OUT eval set
(data/eval/, seed 4242, never used during development) and measures:

Headline metrics
  1. decision accuracy      predicted outcome == expected outcome (3-class), with
                            a full confusion matrix and per-class precision/recall
  2. guardrail recall       every expected guardrail/review warning code raised?
                            (per-code breakdown + false-positive referral rate)
  3. RAG retrieval hit-rate labelled decisive policy sections found in the top-k
                            (micro id-level, case-level all-found, and top-1)
  4. explanation faithfulness  deterministic checker: every citation is a
                            verbatim corpus quote, every cited ID was retrieved or
                            decisive, every number in the rationale exists in the
                            computed facts, and at least one citation is present

Diagnostics: transaction-categorisation accuracy vs generator ground truth
(incl. income-inflation "critical errors"), structured-output success/repair/
fallback rates (LLM mode), per-stage latency, and confidence calibration.

Label caveat (documented in docs/EVALUATION.md): expected outcomes/warnings are
set by scenario *intent* and verified consistent with the rules engine on
ground-truth categories — so decision accuracy measures pipeline integrity and
categorisation robustness, NOT generalisation to real-world data.

Optionally, an LLM judge (same interface) can score rationale groundedness for
comparison with the deterministic checker (`judge_llm=`).
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

from agent import assess, build_facts, validate_rationale
from config import retrieval_config, thresholds
from prompts import PROMPT_VERSION
from schemas import DEBT, AffordabilityMetrics, Outcome

# Committed-credit categories, as strings. A repayment misclassified out of this
# set silently lowers DTI, which is the "debt deflation" critical error class.
DEBT_CATEGORIES = {c.value for c in DEBT}

CLASSES = [Outcome.approve.value, Outcome.refer.value, Outcome.decline.value]
SAFETY = "safety"  # guardrail/review-severity expected warnings
TREATMENT = "info"  # informational data-treatment expected warnings


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(int(round(p * (len(s) - 1))), len(s) - 1)
    return round(s[i], 2)


def _check_faithfulness(decision, diag, retriever, cfg) -> tuple[bool, list[str]]:
    problems: list[str] = []
    # rebuild the exact fact set the rationale was allowed to use
    m_facts = build_facts(
        AffordabilityMetrics(**diag["metrics"]),
        decision.requested_amount,
        decision.term_months,
        decision.monthly_repayment,
        decision.disposable_after_repayment,
        decision.dti_including_new,
        decision.max_affordable_amount,
        categorize_meta=diag.get("categorize_meta"),
    )
    # 1. at least one citation
    if not decision.policy_citations:
        problems.append("no policy citation")
    # 2. citations resolve to the corpus and quotes are verbatim
    for c in decision.policy_citations:
        chunk = retriever.by_id.get(c.policy_id)
        if chunk is None:
            problems.append(f"citation {c.policy_id} not in corpus")
        elif c.quote not in chunk["body"]:
            problems.append(f"citation {c.policy_id} quote is not verbatim")
    # 3. rationale text: cited ids subset of retrieved+decisive, numbers grounded
    allowed_ids = set(diag["retrieved_ids"]) | set(diag["decisive_ids"])
    ok, probs = validate_rationale(decision.rationale, m_facts, allowed_ids, cfg)
    if not ok:
        problems += probs
    return (not problems), problems


def _judge(judge_llm, decision) -> bool:
    system = (
        "You are an evaluator. Answer strictly YES or NO. Is the rationale "
        "fully grounded in the cited policy and the stated metrics, with no "
        "unsupported claims?"
    )
    user = (
        f"Outcome: {decision.outcome.value}\nRationale: {decision.rationale}\n"
        f"Cited: {[c.policy_id for c in decision.policy_citations]}"
    )
    r = judge_llm.chat(system, user)
    return r.text.strip().upper().startswith("YES")


def evaluate(
    applicants: list[dict],
    tx_by: dict[str, list[dict]],
    retriever,
    llm=None,
    truth: dict[str, str] | None = None,
    judge_llm=None,
) -> dict:
    cfg = thresholds()
    rows, faith_viol = [], []
    confusion: dict[tuple, int] = defaultdict(int)
    conf_correct, conf_wrong = [], []
    stage_lat: dict[str, list[float]] = defaultdict(list)
    guard_hits: Counter = Counter()
    guard_expected: Counter = Counter()
    treat_hits = treat_expected = 0
    fp_referrals = approve_expected = 0
    ret_hit = ret_total = case_hits = top1_hits = 0
    rt_hit = rt_total = rt_case_hits = 0  # runtime-decisive coverage
    faithful_n = judge_yes = judge_n = rationale_rejected_n = rationale_llm_n = 0
    cat_correct = cat_total = 0
    cat_conf: Counter = Counter()
    per_cat: dict[str, Counter] = defaultdict(Counter)
    critical_errors = []
    so = Counter()  # structured-output aggregates

    for a in applicants:
        aid = a["applicant_id"]
        decision, diag = assess(a, tx_by[aid], retriever, llm=llm)
        pred, exp = decision.outcome.value, a["expected_outcome"]
        ok = pred == exp
        confusion[(exp, pred)] += 1
        (conf_correct if ok else conf_wrong).append(decision.confidence)

        # guardrail / warning recall (safety codes) + treatment checks (info codes)
        raised = {w.code.value for w in decision.warnings}
        info_codes = {
            "duplicates_removed",
            "internal_transfers_netted",
            "refunds_netted",
            "llm_output_repaired",
            "rationale_rejected",
        }
        exp_safety = [w for w in a["expected_warnings"] if w not in info_codes]
        exp_treat = [w for w in a["expected_warnings"] if w in info_codes]
        for w in exp_safety:
            guard_expected[w] += 1
            guard_hits[w] += int(w in raised)
        treat_expected += len(exp_treat)
        treat_hits += sum(w in raised for w in exp_treat)
        if exp == Outcome.approve.value:
            approve_expected += 1
            fp_referrals += int(pred == Outcome.refer.value)

        # retrieval vs labelled decisive sections
        expected_ids = a["expected_policy_ids"]
        retrieved_ranked = diag["retrieval"]["ids"]
        ret_total += len(expected_ids)
        hits_here = sum(pid in retrieved_ranked for pid in expected_ids)
        ret_hit += hits_here
        case_hits += int(hits_here == len(expected_ids) and expected_ids != [])
        top1_hits += int(
            bool(retrieved_ranked) and expected_ids and retrieved_ranked[0] in expected_ids
        )
        # runtime-decisive coverage: the policy IDs the rules engine ACTUALLY used
        # for this case (diag order matches decisive_ids), found in the top-k
        rt_flags = diag["retrieval_hits"]
        rt_total += len(rt_flags)
        rt_hit += sum(rt_flags)
        rt_case_hits += int(all(rt_flags)) if rt_flags else 0

        # faithfulness
        faith, probs = _check_faithfulness(decision, diag, retriever, cfg)
        faithful_n += int(faith)
        # In LLM mode a rejected OR unattempted rationale is replaced by the
        # deterministic template, which passes the grounding check by construction.
        # Reporting the pass rate alone flatters the model: the 2026-08-01 run had 28
        # of 53 rationales rejected, and the 2026-08-02 provider-outage run had zero
        # model rationales at all yet still scored 0.969. Count what was actually
        # USED (diag.rationale_source), not merely what was not rejected — inferring
        # acceptance from the absence of a rationale_rejected warning reported 32/32
        # accepted during a total provider outage.
        rationale_llm_n += int(diag.get("rationale_source") == "llm")
        rationale_rejected_n += int("rationale_rejected" in raised)
        if not faith:
            faith_viol.append({"applicant_id": aid, "problems": probs[:3]})
        if judge_llm is not None:
            judge_n += 1
            judge_yes += int(_judge(judge_llm, decision))

        # categorisation vs ground truth
        case_cat_ok = case_cat_n = 0
        if truth:
            for tid, pred_cat in diag["categories"].items():
                true_cat = truth.get(tid)
                if true_cat is None:
                    continue
                cat_total += 1
                case_cat_n += 1
                per_cat[true_cat]["support"] += 1
                if pred_cat == true_cat:
                    cat_correct += 1
                    case_cat_ok += 1
                    per_cat[true_cat]["correct"] += 1
                else:
                    cat_conf[(true_cat, pred_cat)] += 1
                    # Two critical error classes — both move an applicant toward a
                    # MORE permissive outcome, which is the direction that matters.
                    #  * income inflation: a non-income credit counted as income
                    #    (numerator of affordability up).
                    #  * debt deflation: a committed repayment counted as ordinary
                    #    spend, so it leaves existing_debt_repayments and DTI falls.
                    # Debt deflation was added after the live-LLM run of 2026-08-01,
                    # where the ONLY unsafe decision (EVAL-026, refer -> approve with
                    # dti_borderline never firing) was caused by lost loan repayments
                    # and was invisible to the income-inflation counter.
                    kind = None
                    if pred_cat == "income" and true_cat not in ("income", "benefits"):
                        kind = "income_inflation"
                    elif true_cat in DEBT_CATEGORIES and pred_cat not in DEBT_CATEGORIES:
                        kind = "debt_deflation"
                    if kind:
                        critical_errors.append(
                            {
                                "kind": kind,
                                "applicant_id": aid,
                                "transaction_id": tid,
                                "true": true_cat,
                                "predicted": pred_cat,
                            }
                        )

        # structured-output stats
        cm = diag["categorize_meta"]
        for k in (
            "batches",
            "llm_calls",
            "parse_failures",
            "repairs",
            "repaired_items",
            "fallback_items",
            "invented_ids",
            "unknown_count",
        ):
            so[k] += cm.get(k, 0)
        so["txns"] += len(diag["categories"])
        so["llm_unavailable"] += int(bool(cm.get("llm_unavailable")))

        for stage, ms in diag["timings_ms"].items():
            stage_lat[stage].append(ms)
        stage_lat["end_to_end"].append(sum(diag["timings_ms"].values()))

        rows.append(
            {
                "applicant_id": aid,
                "profile": a["profile"],
                "expected": exp,
                "predicted": pred,
                "correct": ok,
                "guardrail": decision.guardrail.value if decision.guardrail else None,
                "confidence": decision.confidence,
                "expected_warnings": "|".join(a["expected_warnings"]),
                "raised_warnings": "|".join(sorted(raised)),
                "warning_misses": "|".join(w for w in exp_safety if w not in raised),
                "expected_policy_ids": "|".join(expected_ids),
                "retrieved_ids": "|".join(retrieved_ranked),
                "retrieval_case_hit": hits_here == len(expected_ids),
                "faithful": faith,
                "category_accuracy": round(case_cat_ok / case_cat_n, 3) if case_cat_n else None,
                "end_to_end_ms": round(sum(diag["timings_ms"].values()), 1),
            }
        )

    n = len(applicants)
    correct = sum(1 for r in rows if r["correct"])
    per_class = {}
    for c in CLASSES:
        tp = confusion[(c, c)]
        support = sum(confusion[(c, p)] for p in CLASSES)
        predicted = sum(confusion[(e, c)] for e in CLASSES)
        per_class[c] = {
            "support": support,
            "recall": round(tp / support, 3) if support else None,
            "precision": round(tp / predicted, 3) if predicted else None,
        }

    guard_total = sum(guard_expected.values())
    guard_ok = sum(guard_hits.values())

    summary = {
        "dataset": {
            "n_applicants": n,
            "class_counts": dict(Counter(a["expected_outcome"] for a in applicants)),
            "n_transactions": so["txns"],
        },
        "decision": {
            "accuracy": round(correct / n, 3),
            "correct": correct,
            "n": n,
            "confusion": {
                f"{e}->{p}": confusion[(e, p)]
                for e in CLASSES
                for p in CLASSES
                if confusion[(e, p)]
            },
            "per_class": per_class,
            "false_positive_referral_rate": round(fp_referrals / approve_expected, 3)
            if approve_expected
            else None,
        },
        "guardrails": {
            "recall": round(guard_ok / guard_total, 3) if guard_total else None,
            "true_positives": guard_ok,
            "expected_positives": guard_total,
            "per_code": {
                c: {"expected": guard_expected[c], "raised": guard_hits[c]}
                for c in sorted(guard_expected)
            },
            "treatment_checks": {"expected": treat_expected, "raised": treat_hits},
        },
        "retrieval": {
            # label-based: generator-authored expected policy-ID instances
            "hit_rate_at_k": round(ret_hit / ret_total, 3) if ret_total else None,
            "k": retrieval_config().top_k,
            "hits": ret_hit,
            "expected_ids_total": ret_total,
            "case_all_found_rate": round(case_hits / n, 3),
            "top1_relevant_rate": round(top1_hits / n, 3),
            # runtime-based: the IDs the rules engine actually cited as decisive
            "runtime_decisive_hit_rate": round(rt_hit / rt_total, 3) if rt_total else None,
            "runtime_decisive_hits": rt_hit,
            "runtime_decisive_total": rt_total,
            "runtime_case_all_found_rate": round(rt_case_hits / n, 3),
            "embedder": retriever.embedder.name,
        },
        "faithfulness": {
            "rate": round(faithful_n / n, 3),
            "passed": faithful_n,
            "n": n,
            "checker": "deterministic (verbatim quotes + cited-id subset + "
            "number grounding + citation presence)",
            # Read WITH the pass rate: a rationale the model did not supply, or that
            # failed validation, falls back to the deterministic template, which
            # passes by construction. `used` counts model-written rationales that
            # were actually served; the remainder splits into rejected (validation
            # failed) and unavailable/not-attempted.
            "llm_rationale_used": rationale_llm_n if llm is not None else None,
            "llm_rationale_rejected": rationale_rejected_n if llm is not None else None,
            "llm_rationale_unavailable": (
                n - rationale_llm_n - rationale_rejected_n if llm is not None else None
            ),
            "llm_rationale_used_rate": (
                round(rationale_llm_n / n, 3) if llm is not None and n else None
            ),
            "violations": faith_viol[:10],
            "llm_judge": (
                {"rate": round(judge_yes / judge_n, 3), "n": judge_n} if judge_n else None
            ),
        },
        "categorization": (
            {
                "accuracy": round(cat_correct / cat_total, 3),
                "n_labelled": cat_total,
                "unknown_rate": round(so["unknown_count"] / so["txns"], 3) if so["txns"] else None,
                "per_category": {
                    c: {"support": v["support"], "recall": round(v["correct"] / v["support"], 3)}
                    for c, v in sorted(per_cat.items())
                },
                "top_confusions": [
                    {"true": t, "predicted": p, "count": c} for (t, p), c in cat_conf.most_common(8)
                ],
                "critical_income_inflation_errors": sum(
                    1 for e in critical_errors if e["kind"] == "income_inflation"
                ),
                "critical_debt_deflation_errors": sum(
                    1 for e in critical_errors if e["kind"] == "debt_deflation"
                ),
                "critical_errors_total": len(critical_errors),
                "critical_examples": critical_errors[:5],
            }
            if truth
            else None
        ),
        "structured_output": {
            "mode": "llm" if llm is not None else "rules",
            **{
                k: int(so[k])
                for k in (
                    "batches",
                    "llm_calls",
                    "parse_failures",
                    "repairs",
                    "repaired_items",
                    "fallback_items",
                    "invented_ids",
                    "llm_unavailable",
                )
            },
            "batch_first_pass_valid_rate": round(1 - so["parse_failures"] / so["llm_calls"], 3)
            if so["llm_calls"]
            else None,
            "item_fallback_rate": round(so["fallback_items"] / so["txns"], 3)
            if so["txns"]
            else None,
        },
        "latency_ms": {
            stage: {
                "mean": round(statistics.mean(v), 2),
                "p50": _pct(v, 0.50),
                "p95": _pct(v, 0.95),
            }
            for stage, v in sorted(stage_lat.items())
        },
        "calibration": {
            "mean_confidence_correct": round(statistics.mean(conf_correct), 3)
            if conf_correct
            else None,
            "mean_confidence_incorrect": round(statistics.mean(conf_wrong), 3)
            if conf_wrong
            else None,
        },
        "versions": {
            "prompt": PROMPT_VERSION,
            "llm": f"{llm.name}/{llm.model}" if llm else "deterministic",
            "embedder": retriever.embedder.name,
        },
    }
    return {"summary": summary, "rows": rows}
