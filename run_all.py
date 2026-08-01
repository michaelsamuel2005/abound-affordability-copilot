"""End-to-end run: generate data (if absent) -> build index -> assess the dev set
-> evaluate on the HELD-OUT eval set -> write reports.

    python run_all.py                     # offline deterministic (CI-stable)
    python run_all.py --provider ollama   # real LLM via local Ollama
    python run_all.py --embeddings minilm # sentence-transformers retrieval

Outputs (reports/):
    decisions.json      every dev-set decision with metrics + diagnostics
    eval_summary.json   full evaluation summary (the four headline metrics + more)
    eval_rows.csv       per-applicant evaluation rows
    report.html         human-readable report (tiles + cases + eval tables)
    run_manifest.json   versions, config, dataset sizes, durations — reproducibility
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import ENGINE_VERSION, POLICY_VERSION, retrieval_config, thresholds  # noqa: E402
from dataio import DEV_DIR, EVAL_DIR, load_dataset, load_ground_truth  # noqa: E402
from obs import get_logger, log_event  # noqa: E402
from prompts import PROMPT_VERSION  # noqa: E402

log = get_logger("copilot.run")

BADGE = {"approve": "#1a7f37", "refer": "#9a6700", "decline": "#b31d33"}


def ensure_data() -> None:
    for which, d in (("dev", DEV_DIR), ("eval", EVAL_DIR)):
        if not (d / "applicants.csv").exists():
            subprocess.run(
                [sys.executable, str(ROOT / "data" / "generate_data.py"), "--set", which],
                check=True,
            )


# ---------------------------------------------------------------------------
# HTML report (stat tiles + decision cards + eval tables; no charts by design)
# ---------------------------------------------------------------------------

CSS = """
body{font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;margin:0;
     background:#f6f8fa;color:#1f2937}
.wrap{max-width:1080px;margin:0 auto;padding:26px 22px 48px}
h1{color:#0f2b46;margin:0 0 2px;font-size:23px}h2{color:#0f2b46;font-size:17px;margin:26px 0 10px}
.sub{color:#57606a;margin:0 0 18px;font-size:13.5px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0 6px}
.tile{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:13px 15px}
.tile .lab{font-size:11px;text-transform:uppercase;letter-spacing:.4px;color:#57606a}
.tile .val{font-size:22px;font-weight:700;color:#0f2b46;margin-top:3px}
.tile .det{font-size:11.5px;color:#57606a;margin-top:2px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:11px;padding:14px 16px;margin-bottom:12px}
.row{display:flex;justify-content:space-between;align-items:center;gap:10px}
.badge{color:#fff;border-radius:6px;padding:2px 10px;font-weight:700;font-size:12px;
       text-transform:uppercase}
.muted{color:#57606a;font-size:12.5px}.k{color:#0f2b46;font-weight:600}
.warn{color:#9a6700;font-size:12.5px}.guard{color:#b31d33;font-size:12.5px;font-weight:600}
.cite{color:#0969da;font-size:12px}
.reason{font-size:13.5px;margin:8px 0 4px}
table{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff}
td,th{border:1px solid #e8edf2;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f0f3f6;color:#0f2b46}
.ok{color:#1a7f37;font-weight:600}.bad{color:#b31d33;font-weight:600}
"""


def _tiles(s: dict) -> str:
    d, g, r, f = s["decision"], s["guardrails"], s["retrieval"], s["faithfulness"]
    items = [
        (
            "Decision accuracy",
            f"{d['accuracy'] * 100:.0f}%",
            f"{d['correct']}/{d['n']} regression set",
        ),
        (
            "Grounding-check pass (faithfulness)",
            f"{f['rate'] * 100:.0f}%",
            f"{f['passed']}/{f['n']} deterministic checks",
        ),
        (
            "Retrieval hit-rate@" + str(r["k"]) + " (labelled)",
            f"{r['hit_rate_at_k'] * 100:.0f}%",
            f"{r['hits']}/{r['expected_ids_total']} labelled policy-ID instances",
        ),
        (
            "Runtime-decisive coverage@" + str(r["k"]),
            f"{r['runtime_decisive_hit_rate'] * 100:.0f}%",
            f"{r['runtime_decisive_hits']}/{r['runtime_decisive_total']} runtime-decisive IDs",
        ),
        (
            "Expected-warning recall",
            f"{(g['recall'] or 0) * 100:.0f}%",
            f"{g['true_positives']}/{g['expected_positives']} expected warnings",
        ),
    ]
    cat = s.get("categorization")
    if cat:
        items.append(
            (
                "Categorisation accuracy",
                f"{cat['accuracy'] * 100:.0f}%",
                f"{cat['n_labelled']:,} labelled txns",
            )
        )
    items.append(
        (
            "Pipeline p95 (in-process)",
            f"{s['latency_ms']['end_to_end']['p95']:.0f} ms",
            s["versions"]["llm"],
        )
    )
    return "".join(
        f'<div class="tile"><div class="lab">{escape(k)}</div>'
        f'<div class="val">{escape(v)}</div><div class="det">{escape(det)}</div></div>'
        for k, v, det in items
    )


def _cards(decisions: list[dict]) -> str:
    out = []
    for d in decisions:
        m = d["metrics"]
        cites = " ".join(
            f'<span class="cite">[{escape(c["policy_id"])}]</span>' for c in d["policy_citations"]
        )
        warns = "; ".join(
            f"{w['code']}" for w in d["warnings"] if w["severity"] in ("guardrail", "review")
        )
        guard = (
            f'<span class="guard">guardrail: {escape(d["guardrail"])}</span> · '
            if d.get("guardrail")
            else ""
        )
        meta = (
            f"requested £{d['requested_amount']:,.0f}/{d['term_months']}m · "
            f"repayment £{d['monthly_repayment']:,.0f} · "
            f"assessed income £{m['monthly_income_assessed']:,.0f} · "
            f"disposable £{m['disposable_income']:,.0f} · "
            f"after-repayment £{d['disposable_after_repayment']:,.0f} · "
            f"DTI {d['dti_including_new'] * 100:.0f}% · "
            f"max affordable £{d['max_affordable_amount']:,.0f}"
        )
        out.append(f"""
    <div class="card"><div class="row">
      <div><span class="k">{escape(d["applicant_id"])}</span>
           <span class="muted"> · {escape(d["profile"])}</span></div>
      <span class="badge" style="background:{BADGE[d["outcome"]]}">{escape(d["outcome"])}</span></div>
      <div class="reason">{escape(d["rationale"])}</div>
      <div class="muted">{meta}</div>
      <div style="margin-top:4px">{guard}<span class="warn">{escape(warns)}</span>
        &nbsp; policy: {cites}</div>
    </div>""")
    return "".join(out)


def _eval_table(rows: list[dict]) -> str:
    head = (
        "<tr><th>applicant</th><th>profile</th><th>expected</th><th>predicted</th>"
        "<th>guardrail</th><th>warning misses</th><th>retrieval</th>"
        "<th>faithful</th><th>ms</th></tr>"
    )
    body = []
    for r in rows:
        okc = "ok" if r["correct"] else "bad"
        body.append(
            f"<tr><td>{escape(r['applicant_id'])}</td><td>{escape(r['profile'])}</td>"
            f'<td>{escape(r["expected"])}</td><td class="{okc}">{escape(r["predicted"])}</td>'
            f"<td>{escape(str(r['guardrail'] or ''))}</td>"
            f"<td>{escape(r['warning_misses'] or '—')}</td>"
            f"<td>{'hit' if r['retrieval_case_hit'] else '<span class=bad>miss</span>'}</td>"
            f"<td>{'yes' if r['faithful'] else '<span class=bad>no</span>'}</td>"
            f"<td>{r['end_to_end_ms']:.0f}</td></tr>"
        )
    return f"<table>{head}{''.join(body)}</table>"


def build_html(decisions: list[dict], ev: dict, out: Path) -> None:
    s = ev["summary"]
    conf = s["decision"]["confusion"]
    conf_rows = "".join(
        f"<tr><td>{escape(k)}</td><td>{v}</td></tr>" for k, v in sorted(conf.items())
    )
    cat = s.get("categorization") or {}
    cat_rows = "".join(
        f"<tr><td>{escape(c)}</td><td>{v['support']}</td><td>{v['recall'] * 100:.0f}%</td></tr>"
        for c, v in (cat.get("per_category") or {}).items()
    )
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Affordability Copilot — evaluation report</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>Affordability Copilot — cashflow-underwriting evaluation</h1>
<p class="sub">Held-out eval set: {s["dataset"]["n_applicants"]} applicants /
{s["dataset"]["n_transactions"]:,} transactions · mode: {escape(s["versions"]["llm"])} ·
embeddings: {escape(s["versions"]["embedder"])} · prompts {escape(s["versions"]["prompt"])} ·
engine {ENGINE_VERSION} · policy {POLICY_VERSION}. The LLM never makes the credit
decision — a deterministic rules engine does; recommendations are for human review.</p>
<div class="tiles">{_tiles(s)}</div>
<h2>Dev-set decisions (illustrative cases)</h2>
{_cards(decisions)}
<h2>Evaluation rows (separately seeded regression set)</h2>
{_eval_table(ev["rows"])}
<h2>Confusion (expected → predicted)</h2>
<table><tr><th>transition</th><th>count</th></tr>{conf_rows}</table>
<h2>Categorisation recall per true category</h2>
<table><tr><th>category</th><th>support</th><th>recall</th></tr>{cat_rows}</table>
<p class="muted">Synthetic, seeded data — no real customer data anywhere. Labels are
scenario-intent labels verified against the rules engine (see docs/EVALUATION.md for
the circularity caveat). Set LLM_PROVIDER=ollama|openai|anthropic for LLM
categorisation + generated rationales; EMBEDDINGS=minilm for sentence-transformer
retrieval.</p>
</div></body></html>"""
    out.write_text(html)


# ---------------------------------------------------------------------------


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, help="ollama|openai|anthropic")
    ap.add_argument("--embeddings", default=None, help="tfidf|minilm")
    ap.add_argument(
        "--judge",
        action="store_true",
        help="also run LLM-as-judge on rationales (needs a provider)",
    )
    args = ap.parse_args()
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.embeddings:
        os.environ["EMBEDDINGS"] = args.embeddings

    from agent import assess  # import after env is final  # noqa: E402
    from evaluate import evaluate  # noqa: E402
    from llm import get_llm  # noqa: E402
    from retriever import PolicyRetriever  # noqa: E402

    t_start = time.perf_counter()
    ensure_data()
    llm = get_llm()
    retriever = PolicyRetriever()
    log_event(
        log,
        "run.start",
        llm=llm.name if llm else "deterministic",
        embeddings=retriever.embedder.name,
    )

    # ---- dev set: full decisions for the report ----
    apps, tx_by = load_dataset(DEV_DIR)
    decisions = []
    for a in apps:
        d, diag = assess(a, tx_by[a["applicant_id"]], retriever, llm=llm)
        decisions.append(
            {
                **d.model_dump(mode="json"),
                "profile": a["profile"],
                "metrics": diag["metrics"],
                "retrieved_ids": diag["retrieved_ids"],
                "timings_ms": diag["timings_ms"],
            }
        )

    # ---- held-out eval set ----
    eapps, etx = load_dataset(EVAL_DIR)
    truth = load_ground_truth(EVAL_DIR)
    ev = evaluate(
        eapps, etx, retriever, llm=llm, truth=truth, judge_llm=llm if args.judge else None
    )

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "decisions.json").write_text(json.dumps(decisions, indent=2, default=str))
    (reports / "eval_summary.json").write_text(json.dumps(ev["summary"], indent=2, default=str))
    import pandas as pd

    pd.DataFrame(ev["rows"]).to_csv(reports / "eval_rows.csv", index=False)
    build_html(decisions, ev, reports / "report.html")

    manifest = {
        "engine": ENGINE_VERSION,
        "policy_corpus": POLICY_VERSION,
        "prompts": PROMPT_VERSION,
        "llm": f"{llm.name}/{llm.model}" if llm else "deterministic",
        "embeddings": retriever.embedder.name,
        "retrieval_k": retrieval_config().top_k,
        "thresholds": thresholds().__dict__,
        "dev_set": {
            "applicants": len(apps),
            "transactions": sum(len(v) for v in tx_by.values()),
            "seed": 11,
        },
        "eval_set": {
            "applicants": len(eapps),
            "transactions": sum(len(v) for v in etx.values()),
            "seed": 4242,
        },
        "python": platform.python_version(),
        "duration_s": round(time.perf_counter() - t_start, 1),
    }
    (reports / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    s = ev["summary"]
    print(
        f"\nmode: {manifest['llm']} | embeddings: {manifest['embeddings']} | "
        f"eval: {len(eapps)} applicants / {s['dataset']['n_transactions']:,} txns"
    )
    print(f"  decision accuracy        {s['decision']['accuracy']:.3f}")
    print(
        f"  guardrail recall         {s['guardrails']['recall']:.3f} "
        f"({s['guardrails']['true_positives']}/{s['guardrails']['expected_positives']})"
    )
    print(f"  RAG hit-rate@{s['retrieval']['k']} labelled {s['retrieval']['hit_rate_at_k']:.3f}")
    print(
        f"  runtime-decisive cover   {s['retrieval']['runtime_decisive_hit_rate']:.3f} "
        f"({s['retrieval']['runtime_decisive_hits']}/{s['retrieval']['runtime_decisive_total']})"
    )
    print(f"  faithfulness             {s['faithfulness']['rate']:.3f}")
    if s.get("categorization"):
        print(
            f"  categorisation accuracy  {s['categorization']['accuracy']:.3f} "
            f"(critical income-inflation errors: "
            f"{s['categorization']['critical_income_inflation_errors']})"
        )
    print(f"  end-to-end p95           {s['latency_ms']['end_to_end']['p95']:.0f} ms")
    print(f"reports -> {reports}/report.html\n")
    return manifest


if __name__ == "__main__":
    main()
