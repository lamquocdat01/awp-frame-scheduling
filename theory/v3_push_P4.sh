#!/usr/bin/env bash
# v3_push_P4.sh -- P4 step 3, run by the AUTHOR (Claude Code could not: `gh` was not logged in on 24-10-2026).
# Creates the public repository, pushes main + the submission tag, adds the "superseded" banner to the old TPDS
# repository (one commit, prepared locally, no history rewrite), then runs C7(a) and rebuilds the package so that
# MANIFEST.md gets its "after push" column.            Usage (Git Bash, from the topic folder):  bash code/v3_push_P4.sh
set -euo pipefail
cd "$(dirname "$0")/.."
TOPIC="$(pwd)"
NEW=/d/Repos/awp-frame-scheduling
OLD=/d/Repos/AwP-frame-scheduling-public-old
TAG=v3.0.1-submission

gh auth status >/dev/null 2>&1 || gh auth login --hostname github.com --git-protocol https --web
git -C "$NEW" rev-parse -q --verify "refs/tags/$TAG" >/dev/null || { echo "tag $TAG missing in $NEW"; exit 1; }
[ -z "$(git -C "$NEW" status --porcelain)" ] || { echo "$NEW has uncommitted changes"; exit 1; }
[ "$(git -C "$NEW" grep -c -e 'lamquocdat''@' | wc -l)" = "0" ] || { echo "e-mail found in working tree"; exit 1; }

# 1) new public repository + push (main and the submission tag only)
gh repo view lamquocdat01/awp-frame-scheduling >/dev/null 2>&1 || \
  gh repo create lamquocdat01/awp-frame-scheduling --public \
    --description "Learning-augmented scheduling under hard activation budgets (v3, EJOR submission): code, derived data, one-command reproduction"
git -C "$NEW" remote get-url origin >/dev/null 2>&1 || \
  git -C "$NEW" remote add origin https://github.com/lamquocdat01/awp-frame-scheduling.git
git -C "$NEW" push origin main
git -C "$NEW" push origin "$TAG"

# 2) old TPDS repository: one README commit (banner "superseded; 3 result files truncated"), no rewrite, no deletion
git -C "$OLD" log --oneline -1
git -C "$OLD" push origin main

# 3) C7(a): anonymous raw fetch at the tag, 100 % must match; then rebuild the package (MANIFEST "after push")
sleep 60   # raw.githubusercontent.com can lag a little after a push
PYTHONUTF8=1 python code/v3_c7a_fetch.py --tag "$TAG"
PYTHONUTF8=1 python code/v3_package.py
echo "done: see submission_package/EJOR/MANIFEST.md (C7(a) column) and C7A_raw_fetch.json"
