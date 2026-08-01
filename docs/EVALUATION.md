# Evaluation

## What was measured — and what was not

**Measured configuration: offline deterministic.** Rules-based categorisation,
TF-IDF retrieval, template rationales, **zero LLM calls**. The Ollama/OpenAI/
Anthropic adapters and the MiniLM embedder are implemented and (for the LLM
contract) exercised by scripted-fake tests, but **no live model or dense
embedder was evaluated** — none of the numbers below is a live-LLM result. Run
`python run_all.py --provider ollama` (and `--embeddings minilm`) to produce
those numbers with the same harness.

## Datasets

| Set | Applicants | Transactions | Seed | Purpose |
|---|---|---|---|---|
| dev (`data/raw/`) | 21 | 3,093 | 11 | built/debugged against; report cards |
| eval (`data/eval/`) | 32 | 4,567 | 4242 | **separately seeded regression/evaluation set** — read only by the harness, but used during development to verify label consistency and to fix the retrieval query and labels (v1→v3), so it is NOT a blind, untouched test set |

Both sets: Jan–Jun 2026 window (6 months; thin-file 1 month; one deliberately
missing month), 21-category ground truth per transaction, expected outcome +
warning codes + decisive policy sections per applicant. Eval profiles are
parameter-jittered (±6%; ±0.5% for the two narrow-band profiles, which also use
non-ambiguous descriptions so numeric noise cannot flip their intended label)
and cover every edge scenario: refunds, own-account transfers, duplicates,
cash-heavy, benefits-reliant, no-recognisable-income, gig volatility, gambling,
distress fees, prompt-injection descriptions, ambiguous acquirer strings. All
**21 categories** (including `internal_transfer`) and all **three data
treatments** (refund netting, transfer exclusion, duplicate removal) have
evaluation support.

**Label circularity — read this first.** Expected outcomes are set by scenario
*intent* and verified consistent with the decision rules applied to ground-truth
categories (`tests/test_evaluate.py`). The evaluation therefore measures
**synthetic pipeline integrity** (does the whole chain — categoriser noise
included — still reach the intended decision?) and **robustness** (do refunds,
transfers, duplicates, injections, ambiguous merchants flip anything?), NOT
generalisation to real-world data and NOT live-LLM behaviour. A production
evaluation would need independently labelled real cases.

## Metric definitions (exact)

- **Decision consistency ("decision accuracy")** = predicted outcome ==
  intent-label / 32, three classes, no partial credit. Reported with confusion
  matrix, per-class precision/recall, and false-positive-referral rate.
- **Expected-warning recall (reported as "guardrail recall")** = raised expected
  warning instances / expected instances. On this set: 17 instances across 12
  distinct codes — 8 instances of strict guardrail-severity codes and 9 of
  review-severity codes. Info-severity data-treatment codes are tracked
  separately ("treatment checks", 3 expected). A warning counts only if its
  exact machine code appears in the decision.
- **Retrieval, two measures.** (a) **Runtime-decisive coverage@4** — the policy
  IDs the rules engine ACTUALLY cited as decisive for each case, found in the
  top-4 (this is the operative measure; 40 ID instances). (b) **Labelled
  hit-rate@4** — generator-authored expected policy-ID *instances* (37; the
  corpus has 17 distinct sections). Also reported: case-level all-found rates
  and top-1 relevance. An earlier revision conflated (a) and (b) and mislabelled
  three decline cases (POL-002/POL-009 expected where DTI precedence made
  POL-003 decisive) — found by external audit; labels corrected AND the query
  builder upgraded (v3: loan-inclusive DTI/buffer signals) so both measures now
  reach 100%.
- **Grounding-check pass rate (reported as "explanation faithfulness")** =
  decisions passing ALL deterministic structural checks: (a) ≥1 citation;
  (b) every citation resolves to the corpus and its quote is a verbatim
  substring; (c) every policy ID in the rationale is in the retrieved ∪
  decisive set (decisive IDs are cited even when retrieval missed them — by
  design); (d) every number in the rationale (after stripping `[POL-xxx]`
  tokens) appears in the computed fact/threshold set. **This does NOT prove**
  semantic entailment, factual correctness of the calculations themselves,
  explanation completeness, usefulness to underwriters, or live-LLM
  faithfulness. An optional LLM-as-judge (`--judge`) exists for comparison; it
  was not run.
- **Categorisation accuracy** = correct categories / 4,567 labelled txns, with
  per-category recall, top confusions, unknown rate, and a **critical error
  count**: `income` predicted for anything that isn't income/benefits.

## Results — offline deterministic configuration (2026-07-30, engine 2.0.0, prompts v3, TF-IDF)

| Metric | Value | Raw counts |
|---|---|---|
| Decision consistency | 1.000 | 32/32 · confusion: approve 11/11, refer 15/15, decline 6/6 · FP referrals 0 |
| Expected-warning recall | 1.000 | 17/17 (8 guardrail + 9 review instances; 12 distinct codes, each 100%) · treatment checks 3/3 |
| Retrieval — runtime-decisive coverage@4 | 1.000 | 40/40 runtime-decisive IDs · case-level 32/32 · labelled instances 37/37 · top-1 relevant 53.1% |
| Grounding-check pass | 1.000 | 32/32, zero violations (percentage forms accepted for ratio facts ONLY — the fact×100 hole an audit found is closed and regression-tested) |
| Categorisation accuracy | 0.959 | 4,567 labelled · all 21 categories supported · unknown rate 4.3% · critical income-inflation errors **0** |
| Pipeline latency (in-process, single assessment — **not HTTP API latency**) | p50 ≈4 ms / p95 ≈7 ms | p50 ms: categorise ≈2 · affordability ≈0.9 · retrieve ≈0.7 · decide ≈0.03 |
| Calibration | mean confidence 0.755 on correct; no incorrect decisions to compare | heuristic confidence, NOT calibrated probability |

Notable honest details:

- **The retrieval iteration.** Query v1 used one blanket phrase for all data
  treatments; on refund cases DQ-002 (transfers) outranked DQ-003 (refunds) →
  35/36 at the time. Query v2 emits one phrase per treatment actually applied →
  perfect on the current set. This fix was made *using* this eval set — one of
  the reasons it is a regression set, not a blind one.
- **Categorisation errors are all abstentions.** Every miss is an ambiguous
  acquirer string (`SQ *…`, `CRV*…`, `PAYPAL *…`, `SUMUP *…`) classified
  `unknown` rather than guessed. Per-category recall: 17 of 21 categories at
  100%; shopping 88.3%, dining 93.9%, transport 93.8%, groceries 92.8%.
- **Adversarial LLM behaviour is bounded, not eliminated.** Three hostile
  patterns are tested end-to-end and all end in refer: (1) everything labelled
  `income` → sign-consistency mass-fallback + `llm_output_invalid`; (2) the
  sign-CONSISTENT attack (credits→income, debits→`internal_transfer`, £0
  expenditure) → DQ-007 `transfer_imbalance` + `no_essential_spend` guardrails;
  (3) debits→`savings` → `no_essential_spend`. Residual risk stated plainly:
  subtler sign-valid misclassification within plausible spending shapes can
  still shift metrics and the recommendation — caught only by measurement
  (critical-error count) and human review. Live-LLM injection behaviour:
  unmeasured.
- **Top-1 relevance is 53.1%** even though top-4 is perfect — general
  affordability sections often outrank the case-specific one (and query v3's
  extra clauses lowered top-1 from 62.5% while fixing the decisive misses:
  a real precision/coverage trade-off worth narrating). Fine at k=4.
- **Structured-output metrics** (batch validity, repair, fallback, invented-ID
  rates) are zero-by-construction in rules mode; the harness reports them when
  a provider is configured. The contract is pinned by 11 scripted-fake tests.

## Reproducibility of these numbers

Metric values reproduce identically on re-run (fixed seeds, deterministic
pipeline); **reports are not byte-identical** because timing and duration fields
vary run to run. Retained evidence in `reports/`: `pytest_coverage.txt` (133
passed, 2 skipped optional MiniLM tests; coverage 94.9%), `coverage.xml`,
`docker_build_sandbox.log` (build tail + RAW boot/probe transcript with HTTP
status codes), `eval_summary.json`, `run_manifest.json`.

## What was NOT measured

Live-LLM anything (accuracy, validation/repair rates, rationale grounding,
injection behaviour, latency, tokens, cost — model registries were unreachable
in the authoring environment); MiniLM retrieval quality/latency; HTTP API
latency and throughput under concurrency; human review time; reviewer
agreement; user satisfaction; commercial impact; fairness across demographic
groups (synthetic data cannot support such a claim).

## Reproduce

```bash
python run_all.py          # regenerates data if absent; writes reports/
pytest --cov               # includes the label-consistency regression tests
RUN_MINILM=1 pytest tests/test_minilm_optional.py   # needs sentence-transformers
```
