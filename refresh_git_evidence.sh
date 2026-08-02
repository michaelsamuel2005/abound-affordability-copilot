#!/usr/bin/env bash
# Run AFTER the final commit + push, so the recorded SHA and Actions run URL
# describe the exact tree in the archive. Also re-runs the hermeticity check
# with a visible pass/fail line (the first collection ran it under double -q,
# which suppresses the summary).
#
#     bash refresh_git_evidence.sh
set -uo pipefail
OUT="reports/evidence"
mkdir -p "$OUT"

# hermeticity, with the summary line visible this time
LLM_BATCH_SIZE=10 LLM_TIMEOUT_S=180 DTI_MAX=0.30 BUFFER_GBP=500 EMBEDDINGS=minilm \
  pytest -p no:cacheprovider --override-ini=addopts= --tb=short \
  > "$OUT/pytest_hostile_env.txt" 2>&1
echo "hermeticity: $(grep -E 'passed|failed' "$OUT/pytest_hostile_env.txt" | tail -1)"

{
  echo "=== remote ==="
  git remote -v 2>&1
  REMOTE=$(git remote get-url origin 2>/dev/null)
  echo "repository URL   : ${REMOTE%.git}"
  echo
  echo "=== HEAD (the tree in this archive) ==="
  echo "commit SHA       : $(git rev-parse HEAD 2>&1)"
  echo "short SHA        : $(git rev-parse --short HEAD 2>&1)"
  echo "branch           : $(git rev-parse --abbrev-ref HEAD 2>&1)"
  git log -1 --format='author           : %an <%ae>%ncommitted         : %cd%nsubject          : %s' --date=iso-strict-local 2>&1
  echo
  echo "=== working tree at collection time ==="
  git status --porcelain 2>&1 | sed 's/^/  /'
  [ -z "$(git status --porcelain 2>/dev/null)" ] && echo "  (clean)"
  echo "  NOTE: reports/evidence/ files are written ABOUT the commit above, so"
  echo "  they are necessarily untracked or modified at the instant they are"
  echo "  produced. Source, tests, docs and the five report directories are"
  echo "  committed at the SHA shown."
  echo
  echo "=== commit history ==="
  git log --oneline 2>&1
  echo
  echo "=== GitHub Actions ==="
  if command -v gh >/dev/null 2>&1; then
    echo "--- all runs ---"
    gh run list --limit 10 \
      --json headSha,conclusion,status,workflowName,createdAt,url \
      --jq '.[] | "\(.createdAt)  \(.workflowName)  \(.conclusion)  sha=\(.headSha)  \(.url)"' 2>&1
    echo
    echo "--- run for HEAD ($(git rev-parse --short HEAD)) ---"
    gh run list --commit "$(git rev-parse HEAD)" --limit 3 \
      --json databaseId,conclusion,status,url \
      --jq '.[] | "run id=\(.databaseId)  status=\(.status)  conclusion=\(.conclusion)\n\(.url)"' 2>&1
    echo
    echo "--- jobs in that run ---"
    RID=$(gh run list --commit "$(git rev-parse HEAD)" --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null)
    [ -n "${RID:-}" ] && gh run view "$RID" --json jobs \
      --jq '.jobs[] | "  \(.name): \(.conclusion)"' 2>&1
  else
    echo "gh CLI not installed — open ${REMOTE%.git}/actions and copy the run URL."
  fi
} > "$OUT/git_and_ci.txt" 2>&1

echo
grep -E "^repository URL|^commit SHA" "$OUT/git_and_ci.txt"
sed -n '/--- run for HEAD/,+3p' "$OUT/git_and_ci.txt"
echo
echo "If the run for HEAD is still 'in_progress', wait for it to finish and re-run this."
