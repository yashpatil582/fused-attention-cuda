#!/usr/bin/env bash
# Track B driver, run from the Mac: push the working tree to the VM, run gpu_run.sh there,
# pull the evidence back. rsync never uses --delete (soft-denied; the VM may hold results
# the Mac does not have yet, and a typo must not wipe a build dir mid-session).
#
#   usage: sync.sh --stage N --gpu a10|a100|4090 [--mode quick|full] [--host gpu] [--sweep] [--gpt]
#
# The uncommitted-iteration path (rsync) is for tight loops; for reviewed runs prefer
# `git push && ssh gpu 'cd ~/fused-attention-cuda && git pull --ff-only'` so the CSV `commit`
# column matches a pushed SHA.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STAGE=1; GPU=a10; MODE=quick; HOST=gpu; EXTRA=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --stage) STAGE=$2; shift 2;;
    --gpu) GPU=$2; shift 2;;
    --mode) MODE=$2; shift 2;;
    --host) HOST=$2; shift 2;;
    --sweep|--gpt) EXTRA+=("$1"); shift;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
REMOTE=~/fused-attention-cuda

echo "== rsync -> $HOST:$REMOTE (no --delete)"
ssh "$HOST" "mkdir -p $REMOTE"
rsync -az --exclude build --exclude .git --exclude .research --exclude __pycache__ --exclude '*.egg-info' ./ "$HOST:$REMOTE/"

echo "== remote gpu_run.sh --stage $STAGE --gpu $GPU --mode $MODE ${EXTRA[*]:-}"
set +e
ssh "$HOST" "cd $REMOTE && bash scripts/gpu_run.sh --stage $STAGE --gpu $GPU --mode $MODE ${EXTRA[*]:-}"
rc=$?
set -e

echo "== rsync <- bench/results and profiles (no --delete)"
mkdir -p bench/results profiles
rsync -az "$HOST:$REMOTE/bench/results/" bench/results/
rsync -az "$HOST:$REMOTE/profiles/" profiles/
echo "== done (remote rc=$rc). Remember: stop the instance when idle > 1 h; log spend in docs/WORKLOG.md"
exit $rc
