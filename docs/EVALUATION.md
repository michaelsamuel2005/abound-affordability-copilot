# Evaluation

## What was measured — and what was not

**Four configurations have been measured with the same harness on the same set.**

| # | Configuration | Categorisation | Retrieval | Evidence |
|---|---|---|---|---|
| 1 | **Offline deterministic (headline)** | rules | TF-IDF | `reports/` |
| 2 | Offline deterministic, dense retrieval | rules | MiniLM | `reports_minilm/` |
| 3 | **Live local LLM** | `llama3.2:3b-instruct-q4_K_M` | TF-IDF | `reports_llm_v3/` |
| 4 | Total provider outage (unintentional, kept deliberately) | model unreachable → rules fallback | TF-IDF | `reports_llm_outage/` |

Configuration 1 is the headline because it is the CI-stable, reproducible one.
Configuration 3 is the interesting one: it is where a real model met the
contracts, and it is what turned three latent defects into fixed code (see
[Defects found by measurement](#defects-found-by-measurement)).

`reports_llm_llama32_3b/` retains the **pre-fix** live run — the one that
exposed those defects. Keeping both, with the commits that separate them, is
what makes "I measured it, found bugs in my own guardrails, fixed them and
re-measured" a checkable claim rather than an assertion.

Still unmeasured: hosted providers (OpenAI/Anthropic — no paid call has ever been
made), load and concurrency, cost per assessment, human judgement of the
explanations, and fairness.

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
generalisation to real-world data. A production evaluation would need
independently labelled real cases.

## Metric definitions (exact)

- **Decision consistency ("decision accuracy")** = predicted outcome ==
  intent-label / 32, three classes, no partial credit. Reported with confusion
  matrix, per-class precision/recall, and false-positive-referral rate.
  **Direction matters more than the rate**: a disagreement that is *stricter*
  than the label is the system behaving conservatively; a disagreement that is
  *more permissive* is a safety failure. Both are counted separately.
- **Expected-warning recall (reported as "guardrail recall")** = raised expected
  warning instances / expected instances. On this set: 17 instances across 12
  distinct codes — 8 instances of strict guardrail-severity codes and 9 of
  review-severity codes. Info-severity data-treatment codes are tracked
  separately ("treatment checks", 3 expected). A warning counts only if its
  exact machine code appears in the decision.
- **Retrieval, two measures.** (a) **Runtime-decisive coverage@4** — the policy
  IDs the rules engine ACTUALLY cited as decisive for each case, found in the
  top-4 (the operative measure). **Its denominator is run-dependent**: 40 in the
  deterministic configurations, 51 in the live-LLM run because more guardrails
  fired and therefore more decisive IDs existed to cover. Only compare it against
  a run with the same denominator. (b) **Labelled hit-rate@4** — generator-authored
  expected policy-ID *instances* (37, fixed denominator; the corpus has 17
  distinct sections) — this is the like-for-like measure across configurations.
  Also reported: case-level all-found rates and top-1 relevance.
- **Grounding-check pass rate (reported as "explanation faithfulness")** =
  decisions passing ALL deterministic structural checks: (a) ≥1 citation;
  (b) every citation resolves to the corpus and its quote is a verbatim
  substring; (c) every policy ID in the rationale is in the retrieved ∪
  decisive set (decisive IDs are cited even when retrieval missed them — by
  design); (d) every number in the rationale (after stripping `[POL-xxx]`
  tokens) appears in the computed fact/threshold set. **This does NOT prove**
  semantic entailment, factual correctness of the calculations, explanation
  completeness, usefulness to underwriters, or that the model wrote a good
  explanation.
  **Always read it alongside `llm_rationale_used_rate`.** A model rationale that
  fails validation is replaced by the deterministic template, which passes by
  construction — so in LLM mode a high pass rate can mean "the model wrote well"
  or "the model was overruled and the template answered". The report splits the
  32 decisions into *served* / *rejected by the validator* / *provider
  unavailable* so the two readings cannot be confused. An optional LLM-as-judge
  (`--judge`) exists for comparison; it has not been run.
- **Categorisation accuracy** = correct categories / 4,567 labelled txns, with
  per-category recall, top confusions, unknown rate, and **two critical error
  classes**, both counted because both push an applicant toward a *more
  permissive* outcome:
  - **income inflation** — `income` predicted for something that is not
    income/benefits, raising the affordability numerator;
  - **debt deflation** — a committed repayment (`loan_repayment`, `bnpl`)
    predicted as anything else, so it leaves `existing_debt_repayments` and DTI
    falls. Added 2026-08-02 after the live run showed the only unsafe decision
    was caused by this class and was invisible to the income-inflation counter.

## Results — 1. offline deterministic, TF-IDF (headline; engine 2.0.0, prompts v3)

| Metric | Value | Raw counts |
|---|---|---|
| Decision consistency | 1.000 | 32/32 · confusion: approve 11/11, refer 15/15, decline 6/6 · FP referrals 0 |
| Expected-warning recall | 1.000 | 17/17 (8 guardrail + 9 review instances; 12 distinct codes, each 100%) · treatment checks 3/3 |
| Retrieval — runtime-decisive coverage@4 | 1.000 | 40/40 · case-level 32/32 · labelled instances 37/37 · top-1 relevant 53.1% |
| Grounding-check pass | 1.000 | 32/32, zero violations — rationales here are deterministic templates, so this is a *structural consistency* result, not a model result |
| Categorisation accuracy | 0.959 | 4,567 labelled · unknown rate 4.3% · **income inflation 0 · debt deflation 0** |
| Pipeline latency (in-process, single assessment — **not HTTP API latency**) | p50 ≈4 ms / p95 ≈6 ms | p50 ms: categorise ≈2 · affordability ≈0.9 · retrieve ≈0.7 · decide ≈0.03 |

Every categorisation error here is an **abstention**: an ambiguous acquirer
string (`SQ *…`, `CRV*…`, `PAYPAL *…`, `SUMUP *…`) classified `unknown` rather
than guessed. Per-category recall: 20 of 21 categories at ≥88%; only `shopping`
sits below 90% (88.3%).

## Results — 2. offline deterministic, MiniLM dense retrieval

Identical decisions, guardrails, grounding and categorisation — retrieval does
not influence the outcome, because decisive sections are cited by the rules
engine whether or not retrieval found them. Only retrieval quality and latency
differ:

| Retrieval measure | TF-IDF | MiniLM (`all-MiniLM-L6-v2`) |
|---|---|---|
| Runtime-decisive coverage@4 (denominator 40) | **1.000 (40/40)** | 0.950 (38/40) |
| Labelled hit-rate@4 (denominator 37) | **1.000 (37/37)** | 0.946 (35/37) |
| Case-level all-found | **1.000** | 0.938 |
| Top-1 relevant | **53.1%** | 43.8% |
| Retrieval latency p50 | **0.67 ms** | 10.69 ms (≈16×) |

**The dense embedder is worse on every retrieval measure and sixteen times
slower.** That was not the expected result — the stated hypothesis before running
it was that MiniLM would improve top-1 ordering — and recording the failed
prediction is the point of having run it.

The explanation is the corpus. Seventeen short chunks of tightly-scoped policy
language, queried with text assembled from the same vocabulary, is the case where
lexical matching is strongest: "buffer after repayment", "debt-to-income limit"
and "affordability assessment" are near-synonymous in general-English embedding
space, so a 384-dimensional general-purpose encoder *blurs* precisely the
distinctions that decide which section is decisive. TF-IDF keeps them apart
because it is sharp about words and indifferent to meaning.

This does not generalise, and the result should not be quoted as "embeddings are
worse". A real policy corpus — thousands of heterogeneous chunks drafted by
different teams over years, queried in language that does not match the source —
is where dense retrieval earns its place, and the production answer is hybrid
(BM25 + dense, reciprocal-rank fusion, cross-encoder rerank), because lexical
search remains unbeatable on exact identifiers such as policy IDs and statutory
references. See D-6.

## Results — 3. live local LLM (2026-08-02)

`ollama/llama3.2:3b-instruct-q4_K_M` (digest `a80c4f17acd5`, Ollama 0.32.5),
running **entirely locally** — no transaction data left the machine and no
hosted API was called. TF-IDF retrieval, `LLM_BATCH_SIZE=10`,
`LLM_TIMEOUT_S=180`. The model categorises transactions and drafts rationales;
it has no vote on the outcome.

| Metric | Value | Raw counts |
|---|---|---|
| Decision consistency | 0.969 | 31/32 · approve 11/11, decline 6/6, refer 14/15 · **one disagreement, in the unsafe direction** (refer → approve) · FP referrals 0 |
| Expected-warning recall | 0.824 | 14/17 — missed `dti_borderline` (0/1) and `income_volatility_high` (0/2) · treatment checks 3/3 |
| Retrieval — runtime-decisive coverage@4 | 0.961 | 49/51 — denominator 51, not 40, because more guardrails fired · case-level 0.938 |
| Retrieval — labelled hit-rate@4 | 0.946 | 35/37 · **top-1 relevant 28.1%** (vs 53.1% deterministic) |
| Grounding-check pass | 1.000 | 32/32, zero violations — **but only 8 of 32 rationales are the model's own**; the validator rejected 24 and the template answered instead (`llm_rationale_used_rate` 0.25) |
| Categorisation accuracy | 0.604 | 4,567 labelled · unknown rate 3.8% · **income inflation 3 · debt deflation 33** |
| Structured-output contract | held | 471 batches / 507 calls · 97.0% first-pass valid · 36 repairs (211 items) · 172 item fallbacks (3.8%) · **22 invented policy IDs rejected** · 0 provider outages |
| End-to-end latency (in-process) | p50 261 s / **p95 307 s** | categorise p50 247 s · rationale p50 12.9 s · every other stage <35 ms |

### What this configuration demonstrates

**It is a model-selection result, not a system failure.** The contracts,
guardrails and fallbacks did their jobs; the model did not meet the bar. Four
findings matter more than the scores.

**1. The model stops abstaining, and that is what breaks it.** The rules
categoriser reaches 95.9% with a 4.3% `unknown` rate, and every one of its errors
is an abstention. The 3B model guesses instead: `unknown` recall collapses to
16.7%, `subscriptions` to 7.3% (it calls them `entertainment` 278 times and
`gambling` 71 times), `dining` to 29.9%. It has a coherent taxonomy; it is simply
not this taxonomy. The 60.4% is the visible symptom — the dangerous part is what
those errors do to the arithmetic.

**2. The only unsafe decision came from debt deflation, and that class was
invisible before this run.** EVAL-026 (`borderline_dti`) was approved where the
label says refer, because `dti_borderline` never fired. The mechanism is worth
stating exactly: 33 committed repayments were categorised as something else, most
often `rent_mortgage`. That does **not** reduce total outgoings — it moves money
from the debt bucket into the essentials bucket — so disposable income, the
buffer check and the affordability calculation are all unaffected and all still
pass. But DTI is debt over income, so the numerator falls, the applicant drops
out of the 40–45% manual-review band, and the case is approved.
**The error is invisible to the affordability maths and visible only in the
ratio — and the one guardrail that would have caught it is the guardrail the
error disabled.** The income-inflation counter scored this run "3 critical
errors" and missed all 33.

Debt deflation is now a counted class. There is deliberately **no runtime
guardrail** for it, because detecting it requires ground truth the system does not
have at decision time; it is caught by measurement and human review, and that
limit is stated rather than papered over.

**3. A confident model suppresses guardrails that measure the categoriser's
uncertainty.** `income_volatility_high` missed twice, on both `irregular_income`
profiles — the model's relabelling smoothed the monthly income series below the
cv threshold. Both cases still ended in `refer` through a different rule, so
defence-in-depth held; but that is depth working, not the guardrail working.

**4. Retrieval quality is downstream of categorisation quality.** Top-1 relevance
fell 53.1% → 28.1% with no change to the retriever, the corpus or k. The query is
built from computed risk signals, so wrong metrics produce a wrong query. Any
claim that "retrieval scores X" is conditional on the categoriser feeding it.

### The grounding number needs its companion

32/32 with zero violations is a real improvement on the pre-fix run's 27/32 —
both number-grounding defects are closed. But 24 of 32 model rationales were
**rejected** by the validator, for invented numbers or missing citations, and
replaced by templates. The accurate sentence is: *the deterministic grounding
check passed on every decision, and it got there by overruling three-quarters of
what the model wrote.* That is the validator earning its keep, and it is why the
served / rejected / unavailable split sits next to the pass rate.

## Results — 4. total provider outage (kept as evidence)

Produced by accident on 2026-08-02: the run used the default model alias
`llama3.2`, which was never pulled locally, so **every** Ollama call failed. It is
retained because it measures the degradation path end to end.

All 32 applicants were **referred**. Zero approvals, zero declines, 100% item
fallback to the rules categoriser, `llm_unavailable` on every case. Against intent
labels that scores 0.469 — but read the confusion matrix rather than the rate:
`approve→refer` 11, `refer→refer` 15, `decline→refer` 6. **The system produced no
wrong decisions; it produced abstentions.** That is what "every technical failure
maps to refer" is supposed to mean, now observed under a real outage rather than
a scripted fake (`tests/test_agent.py` covers the scripted version).

## Defects found by measurement

Every one of these was found by running the system, not by reading it. All are
fixed and regression-tested, and the pre-fix evidence is retained.

1. **The plausibility guardrail assumed complete account coverage.** DQ-007's
   transfer-leg balance check fired on every honest single-account applicant,
   because a transfer to an external savings account has no visible counterpart.
   It referred 7 of 11 approve-labelled cases — a false-positive referral rate of
   **0.636**. Now gated on `n_accounts >= 2`; FP referrals are **0.0**. Residual
   risk is stated in the code: on a single account the leg-balance question is
   unanswerable, so `no_essential_spend`, the critical-error counts and human
   review are the remaining defences for that population.
2. **The number validator could not read negative facts.** The extraction regex
   carried no sign, so a template quoting a negative post-repayment buffer as
   `£-458` yielded `458`, which never matched the allowed form `-458`. Four
   over-indebted and marginal cases failed the grounding check because of a
   defect in the checker. The fix keeps the sign meaningful — a rationale claiming
   a *positive* buffer when the fact is negative is still rejected, and that is a
   separate test.
3. **The fact set was an incomplete enumeration.** Deterministic guardrail
   messages are spliced verbatim into the template rationale, and they quote
   quantities (`internal_transfer_net`/`gross`, and the categorisation fallback
   counters) that `build_facts` did not expose — so the *deterministic template*
   failed the system's own grounding check. `build_facts` now enumerates every
   numeric metric field plus the categorisation counters, so any deterministic
   message is grounded by construction. Percentage forms remain restricted to
   `RATIO_FACTS`, so the earlier audit-found fact×100 hole stays closed.

Two further defects were in the *measurement* rather than the system: the
critical-error counter had no debt-deflation class (§3 above), and the first
version of the rationale-acceptance metric inferred acceptance from the absence
of a `rationale_rejected` warning — which reported "32/32 accepted" during a run
in which the model produced nothing at all. Both fixed.

A sixth is worth its own line because it is a different kind of mistake: the
offline test suite was **not hermetic**. Every threshold is env-overridable by
design, so an exported `LLM_BATCH_SIZE=10` left over from a live run split a
40-item scripted response into four batches and broke a guardrail-precedence
assertion — on a developer machine, while CI stayed green because CI starts from
a clean environment. `tests/conftest.py` now clears all 32 configuration
variables before import; tests that want an override use `monkeypatch.setenv`.

## Adversarial behaviour

Three sign-consistent hostile patterns are tested end-to-end and all end in
refer: (1) everything labelled `income` → sign-consistency mass-fallback +
`llm_output_invalid`; (2) credits→income, debits→`internal_transfer`, £0
expenditure → DQ-007 `no_essential_spend` (plus `transfer_imbalance` where
account coverage permits); (3) debits→`savings` → `no_essential_spend`.

Residual risk, stated plainly: subtler sign-valid misclassification within
plausible spending shapes can still shift the metrics and the recommendation.
**The live run is the proof** — a non-hostile model doing its honest best
produced exactly that failure, and it took an unsafe approval to surface it. It
is caught only by measurement (the two critical-error counts) and human review.
Live-LLM behaviour under deliberate prompt injection: unmeasured.

## Reproducibility

Deterministic configurations reproduce metric values identically on re-run (fixed
seeds, pure functions, pinned corpus); **report files are not byte-identical**,
because timing fields vary. Live-LLM runs are **not** reproducible — the provider
is nondeterministic despite temperature 0, so re-running `--provider ollama` will
move those numbers. Treat configuration 3 as a single observation carrying a
stated model digest, not a stable baseline.

Retained evidence: `reports/` (headline), `reports_minilm/`, `reports_llm_v3/`,
`reports_llm_outage/` and `reports_llm_llama32_3b/` (pre-fix), each with
`eval_summary.json`, `eval_rows.csv` and `run_manifest.json`, plus
`pytest_coverage.txt`, `coverage.xml` and `docker_build_sandbox.log`.

## What was NOT measured

Hosted-provider behaviour (OpenAI/Anthropic — no paid call has ever been made,
total spend £0); live-LLM behaviour under deliberate prompt injection; token
usage and cost per assessment; HTTP API latency and throughput under concurrency;
human review time; reviewer agreement; whether the explanations are *useful* to
an underwriter, as opposed to structurally grounded; commercial impact; fairness
across demographic groups, which synthetic data cannot support.

The live-LLM numbers describe **one model at one size on one run**. They are
evidence about `llama3.2:3b-instruct-q4_K_M`, not about LLM categorisation in
general.

## Reproduce

```bash
python run_all.py                                   # configuration 1 (headline)
python run_all.py --embeddings minilm               # configuration 2
LLM_MODEL=llama3.2:3b-instruct-q4_K_M \
  LLM_BATCH_SIZE=10 LLM_TIMEOUT_S=180 \
  python run_all.py --provider ollama               # configuration 3 (~4 h, local)

python inspect_eval.py reports_llm_v3               # confusion direction, missed codes,
                                                    # violations, critical-error examples
pytest --cov                                        # 140 offline tests
RUN_MINILM=1 pytest tests/test_minilm_optional.py   # needs sentence-transformers
```
