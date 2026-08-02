#!/usr/bin/env python3
"""Print the parts of an eval_summary.json that a headline table hides.

Usage:
    python inspect_eval.py                      # reads reports/
    python inspect_eval.py reports_llm_llama32_3b
    python inspect_eval.py reports_minilm

Answers the four questions a summary line cannot:
  1. Which way did the wrong decisions go? (safe over-referral vs unsafe approval)
  2. WHICH expected warnings failed to fire?
  3. Why did each grounding check fail?
  4. What exactly did the model call "income" that was not income?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SAFE_ORDER = {"decline": 0, "refer": 1, "approve": 2}  # higher = more permissive


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "reports")
    path = root / "eval_summary.json"
    if not path.exists():
        print(f"not found: {path}")
        return 1
    s = json.loads(path.read_text())

    print(f"=== {path} ===")
    print("config:", s.get("versions"))
    print("dataset:", s.get("dataset"))

    # ------------------------------------------------------------ decisions
    d = s["decision"]
    print(f"\n== DECISIONS == {d['accuracy']}  ({d['correct']}/{d['n']})")
    unsafe = safe = 0
    for key, count in sorted(d["confusion"].items()):
        expected, predicted = key.split("->")
        if expected == predicted:
            tag = "correct"
        elif SAFE_ORDER[predicted] > SAFE_ORDER[expected]:
            tag = "*** UNSAFE (more permissive than the label) ***"
            unsafe += count
        else:
            tag = "conservative (stricter than the label)"
            safe += count
        print(f"  {key:<22} {count:>3}   {tag}")
    print(f"  totals: {unsafe} unsafe, {safe} conservative")
    print("  false-positive referral rate:", d.get("false_positive_referral_rate"))
    for cls, v in (d.get("per_class") or {}).items():
        print(f"    {cls:<10} {v}")

    # ----------------------------------------------------------- guardrails
    g = s["guardrails"]
    print(f"\n== GUARDRAILS == {g['recall']}  ({g['true_positives']}/{g['expected_positives']})")
    missed = [(c, v) for c, v in g["per_code"].items() if v["raised"] < v["expected"]]
    if not missed:
        print("  every expected warning fired")
    for code, v in missed:
        print(f"  *** MISSED  {code:<28} raised {v['raised']} of {v['expected']}")
    print("  treatment checks:", g.get("treatment_checks"))

    # ---------------------------------------------------------- retrieval
    r = s["retrieval"]
    print(f"\n== RETRIEVAL == embedder={r.get('embedder')} k={r.get('k')}")
    print(
        f"  runtime-decisive {r.get('runtime_decisive_hit_rate')} "
        f"({r.get('runtime_decisive_hits')}/{r.get('runtime_decisive_total')})"
        f"   case-level {r.get('runtime_case_all_found_rate')}"
    )
    print(
        f"  labelled         {r.get('hit_rate_at_k')} "
        f"({r.get('hits')}/{r.get('expected_ids_total')})"
        f"   top-1 relevant {r.get('top1_relevant_rate')}"
    )

    # -------------------------------------------------------- faithfulness
    f = s["faithfulness"]
    print(f"\n== GROUNDING CHECK == {f['rate']}  ({f['passed']}/{f['n']})")
    if f.get("llm_rationale_rejected") is not None:
        print(
            f"  LLM rationales accepted: {f.get('llm_rationale_accepted_rate')} "
            f"({f['n'] - f['llm_rationale_rejected']}/{f['n']}) — the rest fell back "
            f"to the deterministic template, which passes by construction"
        )
    if not f.get("violations"):
        print("  no violations recorded")
    for v in f.get("violations", []):
        print(f"  {v.get('applicant_id')}")
        for p in v.get("problems", []):
            print(f"      - {p}")

    # ------------------------------------------------------ categorisation
    c = s.get("categorization")
    if c:
        print(
            f"\n== CATEGORISATION == {c['accuracy']}  "
            f"({c['n_labelled']} labelled, unknown rate {c.get('unknown_rate')})"
        )
        print(f"  CRITICAL income inflation: {c['critical_income_inflation_errors']}")
        if "critical_debt_deflation_errors" in c:
            print(f"  CRITICAL debt deflation:   {c['critical_debt_deflation_errors']}")
            print(f"  CRITICAL total:            {c.get('critical_errors_total')}")
        else:
            print("  (debt-deflation class not present — pre-fix report)")
        for e in c.get("critical_examples", []):
            print(f"      {e}")
        print("  worst confusions (true -> predicted):")
        for x in (c.get("top_confusions") or [])[:8]:
            print(f"      {x['true']:<18} -> {x['predicted']:<18} {x['count']}")
        weak = sorted(
            ((k, v) for k, v in (c.get("per_category") or {}).items() if v["recall"] < 0.9),
            key=lambda kv: kv[1]["recall"],
        )
        if weak:
            print("  categories below 90% recall:")
            for k, v in weak:
                print(f"      {k:<18} recall {v['recall']:<6} support {v['support']}")

    # ---------------------------------------------------- structured output
    so = s["structured_output"]
    print(f"\n== STRUCTURED OUTPUT == mode={so.get('mode')}")
    for k in (
        "batches",
        "llm_calls",
        "parse_failures",
        "repairs",
        "repaired_items",
        "fallback_items",
        "invented_ids",
        "llm_unavailable",
    ):
        print(f"  {k:<18} {so.get(k)}")
    print("  batch first-pass valid rate:", so.get("batch_first_pass_valid_rate"))
    print("  item fallback rate:         ", so.get("item_fallback_rate"))

    # --------------------------------------------------------------- timing
    print("\n== LATENCY (ms) ==")
    for stage, v in s["latency_ms"].items():
        p95 = v["p95"]
        extra = f"  ({p95 / 1000:.1f} s)" if p95 and p95 > 1000 else ""
        print(f"  {stage:<22} p50 {v['p50']:>12}   p95 {p95:>12}{extra}")

    print("\n== CALIBRATION ==", s.get("calibration"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
