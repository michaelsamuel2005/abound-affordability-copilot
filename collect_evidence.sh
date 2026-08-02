#!/usr/bin/env bash
# Collect the evidence an external auditor asked for, into reports/evidence/.
# Run from the repository root, inside the `abound` conda environment:
#
#     bash collect_evidence.sh
#
# Everything it writes is a real command transcript — nothing is summarised or
# retyped. Takes ~5 minutes, most of it the Docker build.
set -uo pipefail

OUT="reports/evidence"
mkdir -p "$OUT"
echo "writing evidence to $OUT/"

# --------------------------------------------------------------------------
# 0. Environment provenance
# --------------------------------------------------------------------------
{
  echo "=== collected ==="
  date -u "+%Y-%m-%dT%H:%M:%SZ (UTC)"
  date "+%Y-%m-%dT%H:%M:%S%z (local)"
  echo
  echo "=== host hardware (the machine that produced the 307 s live-LLM p95) ==="
  system_profiler SPHardwareDataType 2>/dev/null \
    | sed -n '/Hardware Overview/,/^$/p' \
    | grep -v -i "serial\|hardware uuid\|provisioning udid"
  echo "cpu.brand_string : $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo n/a)"
  echo "physical cores   : $(sysctl -n hw.physicalcpu 2>/dev/null || echo n/a)"
  echo "logical cores    : $(sysctl -n hw.logicalcpu 2>/dev/null || echo n/a)"
  echo "memory (bytes)   : $(sysctl -n hw.memsize 2>/dev/null || echo n/a)"
  echo "page size        : $(sysctl -n hw.pagesize 2>/dev/null || echo n/a)"
  echo
  echo "=== os / runtime ==="
  sw_vers 2>/dev/null
  echo "uname            : $(uname -a)"
  echo "python           : $(python -V 2>&1)"
  echo "python path      : $(which python)"
  echo "conda env        : ${CONDA_DEFAULT_ENV:-<none>}"
  echo "ruff             : $(ruff --version 2>&1 || echo n/a)"
  echo "pytest           : $(pytest --version 2>&1 | head -1 || echo n/a)"
  echo
  echo "=== model serving (live-LLM configuration) ==="
  echo "ollama version   : $(ollama --version 2>&1 || echo 'not installed')"
  echo "-- ollama list --"
  ollama list 2>&1 || echo "n/a"
  echo
  echo "NOTE ON THE 307 s FIGURE: end-to-end p95 for ONE applicant"
  echo "(~145 transactions) with LLM_BATCH_SIZE=10 and LLM_TIMEOUT_S=180, model"
  echo "served locally by Ollama on the CPU/GPU of the machine described above."
  echo "It is in-process pipeline latency, NOT HTTP API latency, and it is"
  echo "dominated by the model: categorise p50 247 s, rationale p50 12.9 s, every"
  echo "deterministic stage under 35 ms. The same machine idle vs in-use moved p95"
  echo "from 470 s (2026-08-01, machine in use) to 307 s (2026-08-02, overnight),"
  echo "so treat these as order-of-magnitude, not benchmark-grade, numbers."
} > "$OUT/environment.txt" 2>&1
echo "  [1/5] environment.txt"

# --------------------------------------------------------------------------
# 1. Tests + coverage (regenerated; the retained copies predate the fixes)
# --------------------------------------------------------------------------
pytest --cov --cov-report=xml:reports/coverage.xml --cov-report=term-missing \
  > reports/pytest_coverage.txt 2>&1
TEST_RC=$?
cp reports/pytest_coverage.txt "$OUT/pytest_coverage.txt"
{
  echo "=== pytest exit code: $TEST_RC ==="
  tail -3 reports/pytest_coverage.txt
} > "$OUT/pytest_summary.txt"
echo "  [2/5] pytest_coverage.txt + coverage.xml (exit $TEST_RC)"

# hermeticity proof: the suite must be immune to ambient config
LLM_BATCH_SIZE=10 LLM_TIMEOUT_S=180 DTI_MAX=0.30 BUFFER_GBP=500 EMBEDDINGS=minilm \
  pytest -q > "$OUT/pytest_hostile_env.txt" 2>&1
echo "  ...  hermeticity check (exit $?) -> pytest_hostile_env.txt"

# --------------------------------------------------------------------------
# 2. Lint (CI runs these)
# --------------------------------------------------------------------------
{
  echo "=== ruff check . ==="; ruff check . 2>&1
  echo; echo "=== ruff format --check . ==="; ruff format --check . 2>&1
} > "$OUT/lint.txt"
echo "  [3/5] lint.txt"

# --------------------------------------------------------------------------
# 3. Docker: canonical python:3.12-slim build, size, inspect, boot + probe
# --------------------------------------------------------------------------
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  {
    echo "=== docker version ==="; docker version 2>&1 | head -12
    echo; echo "=== docker build -t affordability-copilot . (base: python:3.12-slim) ==="
  } > "$OUT/docker_evidence.txt"
  docker build -t affordability-copilot . >> "$OUT/docker_evidence.txt" 2>&1
  BUILD_RC=$?

  {
    echo; echo "=== build exit code: $BUILD_RC ==="
    echo; echo "=== docker images (size on disk) ==="
    docker images affordability-copilot --format \
      "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedSince}}\t{{.Size}}" 2>&1
    echo; echo "=== compressed size (as pushed to a registry) ==="
    docker save affordability-copilot:latest 2>/dev/null | gzip -c | wc -c \
      | awk '{printf "compressed bytes: %d  (%.1f MB)\n", $1, $1/1048576}'
    echo; echo "=== docker image inspect (Config, RootFS, Size) ==="
    docker image inspect affordability-copilot:latest 2>&1
    echo; echo "=== docker history ==="
    docker history affordability-copilot:latest --no-trunc 2>&1
  } >> "$OUT/docker_evidence.txt"

  # boot + probe, raw transcript
  {
    echo; echo "=== container boot + endpoint probe (RAW) ==="
    docker rm -f ac-evidence >/dev/null 2>&1
    echo "\$ docker run -d --name ac-evidence -p 8010:8000 affordability-copilot:latest"
    docker run -d --name ac-evidence -p 8010:8000 affordability-copilot:latest 2>&1
    echo "waiting 15s for uvicorn + HEALTHCHECK..."; sleep 15
    for ep in health ready version; do
      echo "\$ curl -s -i http://localhost:8010/$ep"
      curl -s -i --max-time 10 "http://localhost:8010/$ep" 2>&1 | head -20
      echo
    done
    echo "\$ curl -s -X POST http://localhost:8010/v1/decision  (thin-file applicant)"
    curl -s -i --max-time 30 -X POST "http://localhost:8010/v1/decision" \
      -H 'Content-Type: application/json' \
      -d '{"applicant_id":"EVID-001","requested_amount":3000,"term_months":24,
           "transactions":[{"transaction_id":"TX-1","account_id":"AC-1",
             "date":"2026-06-01","description":"SALARY ACME LTD","amount":2600.0,
             "raw_type":"BGC"}]}' 2>&1 | head -30
    echo
    echo "\$ docker inspect --format '{{json .State.Health}}' ac-evidence"
    docker inspect --format '{{json .State.Health}}' ac-evidence 2>&1
    echo; echo "\$ docker inspect --format 'User={{.Config.User}}' ac-evidence"
    docker inspect --format 'User={{.Config.User}}' ac-evidence 2>&1
    echo; echo "=== container logs ==="
    docker logs ac-evidence 2>&1 | tail -20
    docker rm -f ac-evidence >/dev/null 2>&1
    echo "container removed"
  } >> "$OUT/docker_evidence.txt" 2>&1
  echo "  [4/5] docker_evidence.txt (build exit $BUILD_RC)"
else
  echo "DOCKER NOT AVAILABLE — start Docker Desktop and re-run this script." \
    > "$OUT/docker_evidence.txt"
  echo "  [4/5] docker SKIPPED — Docker daemon not reachable"
fi

# --------------------------------------------------------------------------
# 4. Git + GitHub Actions provenance
# --------------------------------------------------------------------------
{
  echo "=== remote ==="
  git remote -v 2>&1
  REMOTE=$(git remote get-url origin 2>/dev/null)
  echo "repository URL   : ${REMOTE%.git}"
  echo
  echo "=== HEAD ==="
  echo "commit SHA       : $(git rev-parse HEAD 2>&1)"
  echo "short SHA        : $(git rev-parse --short HEAD 2>&1)"
  echo "branch           : $(git rev-parse --abbrev-ref HEAD 2>&1)"
  git log -1 --format='author           : %an <%ae>%ncommitted (UTC)   : %cd%nsubject          : %s' --date=iso-strict-local 2>&1
  echo
  echo "=== working tree (must be clean for the SHA to mean anything) ==="
  git status --porcelain 2>&1 | sed 's/^/  /'
  [ -z "$(git status --porcelain 2>/dev/null)" ] && echo "  (clean)"
  echo
  echo "=== last 12 commits ==="
  git log --oneline -12 2>&1
  echo
  echo "=== GitHub Actions runs ==="
  if command -v gh >/dev/null 2>&1; then
    gh run list --limit 8 \
      --json databaseId,headSha,conclusion,status,workflowName,createdAt,url \
      --template '{{range .}}{{.createdAt}}  {{.workflowName}}  {{.conclusion}}  sha={{.headSha}}  {{.url}}{{"\n"}}{{end}}' 2>&1
    echo
    echo "--- run for the current HEAD ---"
    gh run list --commit "$(git rev-parse HEAD)" --limit 3 \
      --json databaseId,conclusion,status,url \
      --template '{{range .}}id={{.databaseId}} status={{.status}} conclusion={{.conclusion}} {{.url}}{{"\n"}}{{end}}' 2>&1
  else
    echo "gh CLI not installed — open ${REMOTE%.git}/actions and copy the run URL."
  fi
} > "$OUT/git_and_ci.txt" 2>&1
echo "  [5/5] git_and_ci.txt"

# --------------------------------------------------------------------------
# 5. Evidence manifest: what exists, how big, and its checksum
# --------------------------------------------------------------------------
{
  echo "=== evidence directories ==="
  for d in reports reports_minilm reports_llm_v3 reports_llm_outage reports_llm_llama32_3b; do
    if [ -d "$d" ]; then
      echo; echo "--- $d ---"
      ls -l "$d" | tail -n +2 | awk '{printf "  %10s  %s\n", $5, $NF}'
      if [ -f "$d/eval_summary.json" ]; then
        python - "$d/eval_summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
c = s.get("categorization") or {}
print("  config      :", s["versions"])
print("  decision    :", s["decision"]["accuracy"], f'({s["decision"]["correct"]}/{s["decision"]["n"]})',
      s["decision"]["confusion"])
print("  guardrails  :", s["guardrails"]["recall"],
      f'({s["guardrails"]["true_positives"]}/{s["guardrails"]["expected_positives"]})')
print("  retrieval   : runtime", s["retrieval"].get("runtime_decisive_hit_rate"),
      f'({s["retrieval"].get("runtime_decisive_hits")}/{s["retrieval"].get("runtime_decisive_total")})',
      "| labelled", s["retrieval"].get("hit_rate_at_k"), "| top1", s["retrieval"].get("top1_relevant_rate"))
print("  grounding   :", s["faithfulness"]["rate"],
      "| model rationales served:", s["faithfulness"].get("llm_rationale_used"),
      "rejected:", s["faithfulness"].get("llm_rationale_rejected"))
print("  categorise  :", c.get("accuracy"),
      "| income-inflation", c.get("critical_income_inflation_errors"),
      "| debt-deflation", c.get("critical_debt_deflation_errors"))
print("  e2e p95 ms  :", s["latency_ms"].get("end_to_end", {}).get("p95"))
PY
      fi
    else
      echo; echo "--- $d --- MISSING"
    fi
  done
  echo; echo "=== sha256 of every retained artefact ==="
  find reports reports_minilm reports_llm_v3 reports_llm_outage reports_llm_llama32_3b \
    -type f \( -name '*.json' -o -name '*.csv' -o -name '*.log' -o -name '*.xml' -o -name '*.txt' \) \
    2>/dev/null | sort | xargs shasum -a 256 2>/dev/null
} > "$OUT/evidence_manifest.txt" 2>&1
echo "  ...  evidence_manifest.txt"

echo
echo "DONE. Now check these three lines before sending anything:"
echo "---------------------------------------------------------"
grep -E "passed|failed" "$OUT/pytest_summary.txt" | tail -2
grep -E "^affordability-copilot|compressed bytes" "$OUT/docker_evidence.txt" | head -3
grep -E "^commit SHA|^repository URL" "$OUT/git_and_ci.txt"
echo "---------------------------------------------------------"
echo "If the working tree is not clean, commit and re-run so the SHA is meaningful."
