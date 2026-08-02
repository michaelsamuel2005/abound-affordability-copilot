# Design decisions (mini-ADRs)

Format per decision: **Requirement → Alternatives → Chosen → Why → Disadvantage → Mitigation.**

---

**D-1 · Categorisation: LLM (with rules fallback) vs pure rules vs supervised classifier**
Requirement: map free-text merchant strings to a fixed taxonomy.
Alternatives: keyword rules only; train a supervised classifier; LLM only; hybrid.
Chosen: hybrid — LLM inside a Pydantic contract when configured, transparent keyword rules as default *and* as per-item fallback.
Why: an LLM generalises to unseen merchant strings with no labelled training set (the realistic constraint here — no real labelled bank data exists for this project); rules keep the system deterministic for CI and give a safe floor.
Disadvantage: LLM path adds latency, cost and a new failure mode (invalid output); rules path abstains on ~5% ambiguous merchants.
**Measured 2026-08-02:** a local 3B model scored 60.4% categorisation accuracy against the rules path's 95.9%, produced 3 income-inflation and 33 debt-deflation critical errors (rules: 0 and 0), and added ~307 s p95 per applicant against ~6 ms. At this model size the LLM path is strictly worse on every axis that matters here — accuracy, safety and latency. The failure is specifically that it stops abstaining: `unknown` recall falls to 16.7%, and a guess is more dangerous than a gap.
Mitigation: validate-and-repair loop, invented-ID rejection, per-item rule fallback, `unknown` abstention feeding a guardrail; measured fallback/repair rates in the eval harness.
A supervised classifier is the right *next* step once reviewer corrections accumulate labels — the review API already captures them — and the live result strengthens rather than weakens that case: a small trained classifier would beat a 3B generalist on accuracy AND on the 307-second latency, and it can be made to abstain by construction.

**D-2 · Cashflow maths: deterministic code, never the LLM**
Requirement: reproducible, auditable affordability figures.
Alternatives: ask the model to compute; let the model call a calculator tool; pure code.
Chosen: pure code (integer pence, pure functions).
Why: arithmetic from a language model is unverifiable and non-reproducible; in lending the calculation *is* the regulated substance. Same figures every run, unit-testable at boundaries.
Disadvantage: rigid — new metrics need code changes.
Mitigation: thresholds in env-config; metrics carry evidence IDs so extending is mechanical.

**D-3 · Money as integer pence (Decimal conversion), not floats**
Requirement: totals that survive audit.
Alternatives: floats; `Decimal` everywhere; integer minor units.
Chosen: integer pence internally, floats only at the JSON boundary.
Why: float accumulation drifts (0.1+0.2); integer arithmetic is exact and fast; `to_pence` uses `Decimal(str(x))` so float noise can't leak in.
Disadvantage: conversion discipline needed at every boundary.
Mitigation: conversion isolated in two helpers; tests cover half-penny rounding and float-noise cases.

**D-4 · Grounding: RAG over the policy corpus vs model memory vs hard-coding policy in prompts**
Requirement: recommendations justified by the actual current policy, auditable.
Alternatives: model's parametric memory; paste all policy in every prompt; retrieval.
Chosen: retrieval (FAISS) + deterministic citation assembly.
Why: memory hallucinates and goes stale; full-corpus prompting doesn't scale past toy corpora and hides *which* rule mattered; retrieval gives per-decision provenance (IDs, versions, quotes).
Disadvantage: retrieval can miss (observed: 35/36 before query v2) or return noise.
Mitigation: state-dependent query builder, score floor, `retrieval_empty` guardrail, hit-rate measured in eval; decisive sections are also cited directly by the rules, so a retrieval miss degrades *explanation context*, never decision correctness.

**D-5 · Vector store: FAISS vs pgvector vs Elasticsearch**
Requirement: cosine search over 16–10k policy chunks, offline-friendly.
Chosen: FAISS `IndexFlatIP` (exact, normalised = cosine).
Why: zero infrastructure, exact search at this scale, standard skill; pgvector is the production answer once a Postgres exists (it would also give transactional index updates), Elasticsearch is overkill.
Disadvantage: in-process index — no concurrent writers, rebuilt per process.
Mitigation: corpus-hash manifest + cached embeddings; documented pgvector migration path.

**D-6 · Default embeddings: TF-IDF, with MiniLM optional**
Requirement: CI and evals must run offline and deterministically.
Alternatives: dense sentence embeddings by default; TF-IDF by default; hybrid.
Chosen: TF-IDF default; `all-MiniLM-L6-v2` behind `EMBEDDINGS=minilm`.
Why (as designed): policy language and query language share vocabulary by construction, and TF-IDF needs no model download, no torch, no GPU and no version pinning beyond scikit-learn — the vectors are a pure function of the corpus, which is what keeps CI hermetic and the image small.
**Why (as measured, 2026-08-02):** the A/B was actually run, and the dense embedder is worse on *every* retrieval measure — runtime-decisive coverage 38/40 vs 40/40, labelled hit-rate 35/37 vs 37/37, top-1 relevance 43.8% vs 53.1% — while being ~16× slower (retrieval p50 10.69 ms vs 0.67 ms). The hypothesis stated beforehand was that MiniLM would improve top-1 ordering. It did not, and recording the failed prediction is the point of having run it. The explanation is the corpus: 17 short chunks of tightly-scoped policy language, queried with text assembled from the same vocabulary, is the regime where lexical matching is strongest — "buffer after repayment", "debt-to-income limit" and "affordability assessment" are near-synonymous in general-English embedding space, so a 384-dim general-purpose encoder blurs precisely the distinctions that decide which section is decisive.
Disadvantage: TF-IDF fails on paraphrase, and this result does NOT generalise — it is a property of a small, vocabulary-controlled, single-author corpus. It also flatters the retrieval numbers: 40/40 partly reflects a game rigged in TF-IDF's favour, and volunteering that is more credible than being caught on it.
Mitigation: identical FAISS/cosine path for both; embedder name recorded in every result; both configurations measured and reported (docs/EVALUATION.md §2). At real scale the answer is hybrid — BM25 + dense with reciprocal-rank fusion and a cross-encoder rerank over the top ~50 — because lexical retrieval stays unbeatable on exact identifiers (policy IDs, product codes, statutory references), which is exactly what a cited-decision system queries on.

**D-7 · Chunking: section-aligned (one rule per chunk), no overlap**
Requirement: retrieved unit = citable unit.
Alternatives: fixed-size windows with overlap; per-document; per-section.
Chosen: one chunk per policy section (17 chunks, 30–66 words, heading kept).
Why: policy rules are atomic — windows would split a rule or blend two; citations must point at exactly one section ID.
Disadvantage: assumes well-structured markdown; long sections would need sub-splitting.
Mitigation: corpus authored to keep sections atomic; parser enforces unique IDs.

**D-8 · Structured outputs: Pydantic validation + repair, JSON mode where supported**
Requirement: nothing free-text flows into a financial system.
Chosen: provider JSON mode (Ollama `format=json`, OpenAI `response_format`) + Pydantic `extra="forbid"` models + one repair retry + rule fallback.
Why: schema validation catches invalid enums/IDs/fields regardless of provider; repair-with-exact-error recovers most failures cheaply; fallback guarantees output.
Disadvantage: semantically wrong but schema-valid answers still pass (a valid category can be the wrong category).
Mitigation: sign-consistency is schema-enforced (a debit can never carry `income`); DQ-007 plausibility guardrails (transfer-leg balance, zero-essential-spend) defeat the sign-CONSISTENT attacks too (debits mass-labelled transfer/savings) — three adversarial patterns are tested end-to-end and all end in refer; measured category accuracy vs ground truth; income-inflation counted as a critical error class; human corrections rerun the engine. Residual risk stated plainly: subtler sign-valid misclassification within plausible spending shapes remains possible and is caught only by measurement + human review. **The live run of 2026-08-02 is the proof, and it is not hypothetical:** a non-hostile model doing its honest best produced 0 parse failures, 0 provider outages and 97% first-pass schema validity — and still caused an unsafe approval, by labelling 33 committed repayments as ordinary spending. Schema validity and semantic correctness are genuinely independent properties; only the measured critical-error counts separate them, and the income-inflation counter alone did not (hence the debt-deflation class, D-15).

**D-9 · API: FastAPI, versioned routes, sync handlers**
Requirement: ship the pipeline as a service with typed contracts.
Chosen: FastAPI + `/v1` prefix; sync `def` endpoints on the threadpool.
Why: Pydantic-native request/response validation and free OpenAPI docs; the deterministic path is CPU-bound and the LLM client is blocking — sync-on-threadpool is the simplest correct concurrency model at prototype scale.
Disadvantage: blocking LLM calls limit concurrency.
Mitigation: documented; the async migration (httpx.AsyncClient + async endpoints) is mechanical when needed.

**D-10 · Storage: in-memory dict vs database**
Requirement: assessments retrievable and reviewable within a demo session.
Chosen: in-memory store behind a lock.
Why: keeps the prototype honest — no pretend-durability; the review loop is the point, not persistence.
Disadvantage: restart loses history; single-process only.
Mitigation: interface isolated in one `Store` class → Postgres swap is contained; documented loudly.

**D-11 · Orchestration: fixed Python pipeline vs agent framework (LangGraph/LangChain) vs LLM tool-calling**
Requirement: reliable multi-stage flow in a regulated domain.
Chosen: fixed pipeline with state-dependent routing; the LLM never selects tools.
Why: every extra degree of model freedom is an audit surface; a loan assessment has a known correct order; "agentic" value here is routing + validate-and-repair, not autonomy. (Anthropic's own agents guidance: use the simplest pattern that works.)
Disadvantage: no dynamic replanning; "agent" is partly branding.
Mitigation: honest naming in docs; a bounded tool-calling variant is listed future work and the JD's tool-calling skills are demonstrated via the function-calling-shaped contracts (schemas as tools).

**D-12 · Tests: offline and deterministic, LLM via scripted fakes**
Requirement: CI must not flake when a provider changes or is down.
Chosen: 140 offline tests (plus 2 optional env-gated MiniLM integration tests), zero network; FakeLLM scripts exact model behaviour (malformed JSON, invented IDs, sign-inconsistent categories, sign-valid hostile patterns, mid-sequence outages).
Why: live-model tests conflate "my code broke" with "the model changed"; fakes pin the *contract*.
Disadvantage: cannot catch live-model drift (real regression risk).
Mitigation: documented gap, now partially closed — a live smoke-eval has been **run manually** (`--provider ollama`, results in docs/EVALUATION.md §3) and it immediately surfaced what fakes could not: three defects in my own guardrails and validators, plus an unsafe decision class the metrics were not counting. Automating it on a schedule remains the outstanding work.
A second, sharper lesson from the same week: the offline suite was **not hermetic**. Because every threshold is env-overridable (D-2's mitigation), an exported `LLM_BATCH_SIZE=10` from a live run silently changed what the tests measured and broke a guardrail-precedence assertion on a developer machine while CI stayed green. `tests/conftest.py` now clears all 32 configuration variables before import. Making a system configurable makes its tests configurable too — that is a cost of the flexibility, not a free property.

**D-13 · Outcomes: three classes with refer-as-abstention**
Requirement: an automated recommendation must be allowed to say "a human must look".
Chosen: approve / refer / decline; every guardrail and technical failure maps to refer.
Why: binary approve/decline forces overconfidence; refer is the abstention channel that makes conservative behaviour explicit and measurable (false-positive-referral rate 0 on eval).
Disadvantage: over-referral is operationally expensive at scale.
Mitigation: per-code guardrail metrics expose which rule refers how often.

**D-14 · Human review: functional API, not a UI**
Requirement: prove the loop (see → correct → recompute → audit), not paint it.
Chosen: review endpoints with mandatory reasons, category corrections, deterministic recompute, retained originals.
Disadvantage: no reviewer ergonomics; automation-bias risk unaddressed.
Mitigation: documented; corrections are captured in a form that later trains D-1's classifier.

**D-15 · Evaluation: these four headline metrics**
Requirement: measure the four ways this system can fail a lender — wrong outcome (decision accuracy), unsafe silence (guardrail recall), ungrounded justification (faithfulness), wrong grounding context (retrieval hit-rate).
Chosen: those four + categorisation accuracy, structured-output rates, latency as diagnostics.
**Revised 2026-08-02:** categorisation accuracy now carries TWO critical-error classes, not one. The original counted only income inflation (a non-income credit read as income). The live run showed the only unsafe decision came from the mirror-image error — a committed repayment read as ordinary spending, which leaves total outgoings and disposable income untouched but lowers DTI, dropping a borderline applicant out of the manual-review band. 33 such errors occurred against 3 income-inflation ones; the metric as designed reported "3 critical errors" and missed the class that actually caused harm. Both are now counted. Note the honest limit: debt deflation has **no runtime guardrail**, because detecting it needs ground truth the system lacks at decision time — it is caught by measurement and human review only.
Disadvantage: label circularity (generator intent ↔ rules) makes headline accuracy easy; no human evaluation.
Mitigation: circularity documented prominently; categorisation robustness measured against genuinely ambiguous inputs; live-model runs use the same harness.

**D-16 · Packaging: Docker (single hardened image) vs venv instructions vs Compose stack**
Requirement: anyone — an interviewer, CI, another machine — can run the identical service without inheriting local setup.
Alternatives: README-only venv instructions; docker-compose multi-service; a single container.
Chosen: one non-root container built from `requirements.lock`, seeded demo data regenerated at build time, stdlib health check, port 8000.
Why: exact-pin reproducibility (lockfile → identical dependency tree), parity between CI and local runs, and non-root/healthcheck/EXPOSE is the minimum surface a platform team expects to see.
Disadvantage: image weight (**757 MB on disk / 162 MB compressed**, measured on the canonical `python:3.12-slim` base both locally and in CI) and a false air of deployability — a container is not a deployment.
Mitigation: slim base target + `.dockerignore`; README and docs state loudly that containerised ≠ deployed (see the §22 honesty in FACTS / docs/EVALUATION.md scope notes).
