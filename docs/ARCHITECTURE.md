# Architecture

## One-paragraph summary

A fixed Python pipeline (not an autonomous agent) assesses one loan application
at a time: validate input → categorise transactions (LLM inside a Pydantic
contract, or deterministic rules) → compute affordability metrics in integer
pence → build a retrieval query from those metrics → fetch top-4 policy sections
from a FAISS index → apply deterministic decision rules with guardrails →
assemble citations and a validated rationale → return a recommendation that a
human must review. State lives per-request; the service is stateless between
applications except for the in-memory assessment/review store.

## Stage detail

### 1. Input validation (`schemas.Transaction`)
Every transaction needs a unique `transaction_id`, `account_id`, ISO date,
description (≤200 chars), non-zero ≤2dp GBP amount. `extra="forbid"` everywhere:
unknown fields are rejected, not ignored. Requests are capped
(`MAX_TRANSACTIONS=5000`) and terms bounded (6–84 months, ≤£50k).

### 2. Categorisation (`categorize.py`)
- **Rules path (default / fallback):** ordered keyword rules; sign is not
  category evidence for inflows — refund/transfer/benefit detection runs first
  and an unrecognised credit is `unknown`, never `income` (the costliest error in
  underwriting is inflating income). Unknown → confidence 0.30.
- **LLM path:** batches of 40 transactions serialised as JSON lines inside
  `<transactions>` tags (untrusted-data delimiter). System prompt v3 carries the
  21 category definitions, 7 few-shot examples (including refund / own-account
  transfer / benefits / ATM / ambiguous-abstain), an explicit unknown-abstention
  rule and anti-injection instructions. Reply must parse into
  `CategorizationBatchResult`; duplicate IDs, invalid enums or extra fields fail
  validation → one repair attempt quoting the exact error → per-item fallback to
  rules. Invented IDs are dropped and counted; missing IDs fall back; a
  category inconsistent with the transaction's sign (e.g. `income` on a debit)
  fails schema validation and falls back. The model can therefore *only*
  influence category + confidence (within the correct sign), never amounts,
  IDs or essential/discretionary mapping — a residual risk remains that a
  schema-valid wrong category shifts the metrics, which is why category
  accuracy and income-inflation errors are measured and humans review.

### 3. Affordability engine (`affordability.py`)
Integer-pence arithmetic (`Decimal(str(x)) * 100`), floats only at the boundary.
Passes, in order: exact-duplicate removal → internal-transfer exclusion (both
legs) → refund matching (same account, merchant stem, ≤45 days; matched refunds
net against the original category, unmatched are excluded entirely) → bucket
aggregation. Eligible income = `income` + `benefits` credits only. Volatility =
population CV of monthly eligible income; above 0.35 the assessed income is
`min(worst month, mean × (1 − min(cv, 0.5)))`. Disposable = assessed income −
essentials − existing debt − unknown debits (conservative). Every aggregate
carries its supporting transaction IDs in `metrics.evidence`.

### 4. Retrieval (`retriever.py`)
Corpus: `policy/*.md`, parsed into one chunk per `### [POL-xxx]/[DQ-xxx]`
section (17 chunks, 30–66 words — section-aligned by design; a rulebook's atomic
unit is the rule). Embeddings: TF-IDF (1–2gram, sklearn) by default —
deterministic and offline — or `all-MiniLM-L6-v2` (384-dim) with corpus-hash-keyed
caching. FAISS `IndexFlatIP` over L2-normalised vectors = exact cosine; top-4
with a minimum-score floor (0.05 tfidf / 0.25 minilm); below-floor results are
dropped and an empty result raises the `retrieval_empty` guardrail. The query is
built deterministically from the case's computed signals — including the
LOAN-INCLUSIVE ones (post-repayment buffer, DTI including the new loan; query
v3) and classification-plausibility flags — so retrieval is state-dependent.
A retrieval exception is caught and degrades to the `retrieval_empty` guardrail
(refer), never a 500.

### 5. Decision rules (`decision_rules.py`)
22 fixed warning codes (11 guardrail / 6 review / 5 info), including the DQ-007
classification-plausibility guardrails: internal-transfer legs that do not
broadly balance, or a ≥40-transaction window with zero essential spending, force
a referral — this is what defeats sign-consistent hostile categorisation (e.g.
every debit labelled `internal_transfer` or `savings`).
Documented fixed precedence: pipeline/data guardrails (→ refer) → DTI > 45%
decline → gambling > 10% refer → distress ≥ 2 refer → CV > 0.35 refer → buffer
< £150 (reduced-offer refer if ≥£500 and ≥50% of ask affordable, else decline) →
DTI 40–45% refer → vulnerability (benefits > 50% ∧ essentials > 60%) refer →
approve. Invariants (unit-tested): guardrails and technical failures can never
produce an automatic decline; every outcome carries ≥1 policy ID; thresholds come
from env-overridable config, not code edits.

### 6. Explanation & citations (`agent.py`)
Citations are assembled from corpus metadata — policy ID, doc, version, verbatim
quote — never from model text, so a fabricated citation is structurally
impossible. The optional LLM rationale is validated: cited IDs ⊆ retrieved ∪
decisive, and every number (after stripping `[POL-xxx]` tokens) must match the
computed fact set (with formatting/percentage tolerance). Failures are logged,
flagged (`rationale_rejected`) and replaced by the deterministic template.
Percentage number-forms are accepted for RATIO facts only (a £3,000 income can
never legitimise a fabricated £300,000). Every policy ID the accepted rationale
cites — including retrieved-but-not-decisive ones — gets a structured citation
object. If a guardrail fires AFTER the rules ran (e.g. rationale-stage provider
outage), the whole decision is rewritten coherently: refer outcome, 0.5
confidence, referral-first reasons (original assessment kept as context), and
the guardrail policy cited first.

### 7. Human review (`app.py`)
`human_review_required` is always true — the system recommends, people decide.
Reviewers see metrics, warnings, citations, evidence and (on request) raw
transactions with assigned categories; they uphold or override (mandatory
outcome + ≥10-char reason) and can correct categories, which triggers a
deterministic recompute (LLM never involved). The review record persists the
original recommendation, reviewer identity, reason, the EXACT category
corrections and the full recomputed metrics + outcome (in the in-memory store —
durability is a documented gap).

## Cross-cutting

- **Versioning:** every DECISION object carries engine 2.0.0, policy corpus
  2026-07.x, prompt v3, model id and embedder; every LOG LINE carries the
  engine version and request/applicant IDs, with model + prompt version on the
  per-decision log events — so any result is attributable to an exact
  configuration.
- **Observability:** single-line JSON logs (stdout) with request ID, applicant
  ID, stage, latency; transaction descriptions and names are never logged.
  Honest scope: structured logs + eval reports; no metrics store, traces or
  dashboards.
- **Determinism:** seeds fixed (data 11/4242, LLM seed 42, temperature 0), CI
  runs no network calls; metric values reproduce identically on re-run
  (timing/duration fields naturally vary, so report files are not
  byte-identical).

## What is deliberately NOT here

An agent framework (a fixed pipeline is easier to audit than dynamic tool
selection — see docs/DECISIONS.md D-11), a database (in-memory store keeps the
prototype honest about its scope), authentication, and any real-data ingestion.
