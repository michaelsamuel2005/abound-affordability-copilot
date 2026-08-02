# Evidence pack — index

Every item below is a real command transcript under `reports/evidence/`. Nothing
here is retyped or summarised from memory; regenerate the whole pack with
`bash collect_evidence.sh`.

| # | Requested | Where | Value |
|---|---|---|---|
| 1 | Updated source and tests | `src/`, `tests/`, `app.py`, `run_all.py` | **140 tests pass, 2 skipped** (optional MiniLM), coverage **94.9%**, `ruff check` and `ruff format --check` clean |
| 2 | `reports_minilm/` | present | deterministic + MiniLM |
| 3 | `reports_llm_v3/` | present | live `llama3.2:3b-instruct-q4_K_M` |
| 4 | `reports_llm_outage/` | present | total provider outage |
| 5 | `reports_llm_llama32_3b/` | present | **pre-fix** live run — the one that exposed the defects |
| 6 | Updated coverage | `reports/pytest_coverage.txt`, `reports/coverage.xml` | regenerated against the current tree |
| 7 | All `eval_summary.json` / `eval_rows.csv` / `run_manifest.json` | 5 directories × 3 files | SHA-256 of every artefact in `reports/evidence/evidence_manifest.txt` |
| 8 | Updated `conftest.py` | `tests/conftest.py` | clears 32 config env vars before import (hermeticity) |
| 9 | Repo URL / commit SHA / Actions run | `reports/evidence/git_and_ci.txt` | see below |
| 10 | Docker size + inspection | `reports/evidence/docker_evidence.txt` | see below |
| 11 | Hardware behind the 307 s p95 | `reports/evidence/environment.txt` | see below |

## Measured configurations

All five share one harness, one eval set (32 applicants / 4,567 labelled
transactions, seed 4242), engine 2.0.0, prompts v3.

| Directory | Categoriser | Embedder | Decision | Guardrail recall | Grounding | Categorisation |
|---|---|---|---|---|---|---|
| `reports/` (headline) | rules | tfidf | **1.000** (32/32) | 1.000 (17/17) | 1.000 | 0.959 · 0 + 0 critical |
| `reports_minilm/` | rules | minilm | 1.000 (32/32) | 1.000 (17/17) | 1.000 | 0.959 · 0 + 0 critical |
| `reports_llm_v3/` | llama3.2 3B | tfidf | **0.969** (31/32) | 0.824 (14/17) | 1.000 † | 0.604 · **3 income-inflation + 33 debt-deflation** |
| `reports_llm_outage/` | unreachable → rules | tfidf | 0.469 ‡ | 0.471 | 0.969 | 0.959 |
| `reports_llm_llama32_3b/` (pre-fix) | llama3.2 3B | tfidf | 0.750 (24/32) | 0.824 (14/17) | 0.844 | 0.604 · 3 + (class not yet counted) |

† Read with `llm_rationale_used_rate` = 0.25: only **8 of 32** rationales were the
model's own. The validator rejected 24 and the deterministic template answered,
and the template passes by construction. The pass rate alone would flatter the
model, which is why the served / rejected / unavailable split is reported.

‡ Not a quality figure. Every call to the (un-pulled) `llama3.2` alias failed, so
all 32 applicants were **referred**: `approve→refer` 11, `refer→refer` 15,
`decline→refer` 6. Zero approvals, zero declines — the system produced
abstentions, not wrong decisions. Retained deliberately as evidence of the
degradation path.

The one unsafe decision in `reports_llm_v3` is EVAL-026: 33 committed repayments
were categorised as ordinary spending (mostly `rent_mortgage`), which leaves
total outgoings and disposable income unchanged but lowers DTI, dropping a
borderline applicant out of the 40–45% manual-review band so `dti_borderline`
never fired. Full analysis in `docs/EVALUATION.md`.

## 9 · Repository, commit, CI

```
repository   https://github.com/michaelsamuel2005/abound-affordability-copilot
branch       main
```

**The authoritative SHA and run URL are in `reports/evidence/git_and_ci.txt`,
regenerated after the final commit** — that file is written *about* a commit, so
it is necessarily untracked at the moment it is produced. That is not a dirty
working tree in any meaningful sense; `git_and_ci.txt` records the tree state at
collection time so it can be checked.

CI has been green on every push since the repository was created. At the time of
writing, four runs, four successes:

```
7976d82  Docs: four measured configurations; add evidence collector   success
236e9e0  Ground categorisation counters; fix rationale-acceptance metric  success
096519c  Fix defects found by the live-LLM run; make offline tests hermetic  success
e14830f  Affordability Copilot: human-in-the-loop RAG for cashflow underwriting  success
```

Each run executes: ruff lint + format check, 140 tests on Python 3.11 **and**
3.12 installed from `requirements.lock`, an end-to-end offline evaluation, and a
Docker build-and-boot job probing `/health` and `/ready`. `.git` is included in
this archive so the SHAs can be verified independently.

## 10 · Docker

Built from the canonical `python:3.12-slim` base (build exit code 0).

| Measure | Value | Source |
|---|---|---|
| Image ID | `sha256:a402ad66c4b8…` | `docker image inspect` |
| **Size on disk (unpacked)** | **757 MB** | `docker images` |
| **Compressed (as pushed)** | **161,071,136 bytes ≈ 161 MB** | `docker save \| gzip \| wc -c` |
| `inspect .Size` | 162,327,164 bytes | containerd image store reports the compressed layer sum, which is why it agrees with the gzip figure rather than with 757 MB |
| Platform | `arm64` / `linux` | Apple Silicon host |
| Runtime user | `copilot` (**non-root**) | `.Config.User` |
| Exposed port | `8000/tcp` | `.Config.ExposedPorts` |
| Healthcheck | stdlib `urllib` against `/health` (no curl in slim) | `.Config.Healthcheck` |
| Layers | 13 | `.RootFS.Layers` |

Boot and probe transcript (raw `curl -i`, full response bodies retained):

```
GET  /health        HTTP/1.1 200 OK
GET  /ready         HTTP/1.1 200 OK
GET  /version       HTTP/1.1 200 OK
POST /v1/decision   HTTP/1.1 201 Created
    -> outcome "refer", guardrail "insufficient_history",
       warnings [insufficient_history, low_transaction_count, no_recognisable_income],
       citations [POL-007, DQ-006] with verbatim quotes and retrieval scores
docker inspect .State.Health -> {"Status":"healthy","FailingStreak":0}
docker inspect .Config.User  -> copilot
```

A container is not a deployment: this proves packaging and contract discipline,
not scalability or operation. Nothing here has ever been deployed.

## 11 · Hardware behind the 307-second p95

```
Chip            Apple M1
Cores           8 physical / 8 logical
Memory          8,589,934,592 bytes (8 GB unified)
Machine         MacBook Air (Darwin 25.1.0, arm64, T8103)
OS              macOS 26.1 (build 25B78)
Python          3.12.13 (conda env `abound`)
pytest 9.1.1 · ruff 0.16.1
Ollama          0.32.5
Model           llama3.2:3b-instruct-q4_K_M — ID a80c4f17acd5, 2.0 GB
Settings        LLM_BATCH_SIZE=10, LLM_TIMEOUT_S=180, temperature 0
```

**What the number is, precisely.** 307,147 ms is the p95 of *end-to-end
in-process pipeline latency for one applicant* (~145 transactions) — **not** HTTP
API latency, and not a throughput measurement. It is almost entirely the model:
categorise p50 247 s, rationale p50 12.9 s, and every deterministic stage under
35 ms (affordability 3.6 ms, retrieval 28.9 ms, decision 0.2 ms).

**Treat it as order-of-magnitude, not benchmark-grade.** The same machine and the
same model produced p95 470 s on 2026-08-01 while the machine was in use and
307 s on 2026-08-02 running overnight — a 35% swing from host load alone. No
attempt was made to control for thermal state, background processes or GPU
contention, and no repeated trials were run. A defensible benchmark would need
all three; this figure exists to establish that a local 3B model costs minutes
per applicant rather than milliseconds, which is the decision-relevant fact.

Deliberately not measured, and not estimated: throughput, concurrency, HTTP API
latency under load, cost per assessment (the run was local, so there is no cost),
reviewer usefulness of explanations, and fairness across groups.
