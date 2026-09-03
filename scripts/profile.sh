#!/usr/bin/env bash
# Nsight Compute per stage (judge_plan §4.5). Runs on the GPU box after the bench.
#
#   usage: profile.sh <stage> <gpu>
#
# For every kernel of the stage (scripts/kernel_regex.py, CONTRACT §2) and both canonical
# configs (C64, C128) run ONE ncu pass on the torch-free fa_bench (so only our kernels
# appear), export .ncu-rep, then re-import as a wide raw CSV (`--page raw`: one row per
# launch, one column per metric ID; `--section` is ignored on import, which is what we want).
#
# Never fabricates: if the counters are blocked (ERR_NVGPUCTRPERM) the verbatim error goes
# to profiles/<gpu>/stage<N>.profiler.txt and we fall back to an nsys timeline + stats.
# Metric IDs are pruned ONCE per GPU against `ncu --query-metrics` (dropped IDs logged).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STAGE=${1:?usage: profile.sh <stage> <gpu>}; GPU=${2:?usage: profile.sh <stage> <gpu>}
PY=$(command -v python3 || command -v python)
PROF=profiles/$GPU; OUT=bench/results/$GPU; mkdir -p "$PROF" "$OUT"
PTXT=$PROF/stage${STAGE}.profiler.txt
BENCH=$(ls build/bench/fa_bench build/fa_bench 2>/dev/null | head -1 || true)
[ -n "$BENCH" ] || { echo "profile.sh: fa_bench not built"; exit 1; }
DTYPE=$("$PY" scripts/kernel_regex.py --dtype "$STAGE")
C64=$("$PY" scripts/kernel_regex.py --cfg C64); C128=$("$PY" scripts/kernel_regex.py --cfg C128)
rm -f "$PTXT"
# Provenance (PROFILING_PROTOCOL: "taken at the same commit as its benchmark row"): fa_bench
# stamps FA_COMMIT into its rows, and stage<N>.meta records commit/date/ncu/clock/dtype plus the
# exact ncu command line of every report; ncu_to_md.py prints the commit next to each row and
# tests/test_bench_sanity.py fails when it differs from the stage's bench rows.
if [ -z "${FA_COMMIT:-}" ]; then FA_COMMIT=$("$PY" -c "import sys; sys.path.insert(0,'scripts'); from fa_common import git_commit; print(git_commit())"); fi
export FA_COMMIT
META=$PROF/stage${STAGE}.meta
{
  echo "commit=$FA_COMMIT"; echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "stage=$STAGE"; echo "gpu=$GPU"
  echo "dtype=$DTYPE"; echo "cfg_C64=$C64"; echo "cfg_C128=$C128"; echo "bench=$BENCH"
} > "$META"

record() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$PTXT"; }

nsys_fallback() {
  # Degraded mode: CUDA activity tracing needs no hardware counters (UNVERIFIED on Colab —
  # this is the empirical test). Timeline + per-kernel stats replace ncu; README says so.
  if command -v nsys >/dev/null; then
    if nsys profile -t cuda -o "$PROF/stage${STAGE}_timeline" -f true --stats=true "$BENCH" --stage "$STAGE" --cfg "$C64" --dtype "$DTYPE" --causal 0 --iters 12 > "$PROF/stage${STAGE}_nsys_stats.txt" 2>&1; then
      record "nsys fallback ok: $PROF/stage${STAGE}_timeline.nsys-rep + stage${STAGE}_nsys_stats.txt"
    else record "nsys fallback FAILED: $(tail -1 "$PROF/stage${STAGE}_nsys_stats.txt")"; fi
  else record "nsys missing: no timeline fallback available"; fi
  rm -f "$PROF"/*.sqlite
}

if ! command -v ncu >/dev/null; then
  record "ncu missing on PATH; hardware counters unavailable on this runtime"
  nsys_fallback; exit 0
fi
NCU_VER=$(ncu --version | head -1)
grep -q "ncu_profile_version=" "$OUT/env.txt" 2>/dev/null || echo "ncu_profile_version=$NCU_VER" >> "$OUT/env.txt"
echo "ncu_version=$NCU_VER" >> "$META"

# ---- prune the metric list once per GPU against what this ncu/GPU actually exposes ----
USED=$PROF/ncu_metrics.used.txt
if [ ! -s "$USED" ]; then
  ncu --query-metrics > "$PROF/ncu_query_metrics.txt" 2>&1 || true
  ncu --list-sections > "$PROF/ncu_list_sections.txt" 2>&1 || true
  "$PY" - "$PROF" <<'EOF'
import re, sys
prof = sys.argv[1]
avail = set()
for line in open(f"{prof}/ncu_query_metrics.txt", errors="replace"):
    tok = line.split()
    if tok and re.fullmatch(r"[a-z][a-z0-9]*__[a-z0-9_]+", tok[0]):
        avail.add(tok[0])
used, dropped, enabled = [], [], []
for line in open("scripts/ncu_metrics.txt"):
    s = line.strip()
    unverified = s.startswith("# ") and re.fullmatch(r"# [a-z][a-z0-9_.]+", s) is not None
    if not s or (s.startswith("#") and not unverified):
        continue
    mid = s[2:] if unverified else s
    base = mid.split(".")[0]
    if not avail:  # query failed: keep the verified list, never enable unverified IDs blind
        (used if not unverified else dropped).append(mid)
    elif base in avail or mid.startswith("launch__"):  # launch__* are launch attrs, not in the counter list
        (enabled if unverified else used).append(mid)
    else:
        dropped.append(mid)
with open(f"{prof}/ncu_metrics.used.txt", "w") as f:
    f.write("# pruned from scripts/ncu_metrics.txt against `ncu --query-metrics` on this GPU\n")
    f.write("\n".join(used + enabled) + "\n")
print(f"metrics: {len(used)} kept, {len(enabled)} UNVERIFIED enabled {enabled}, {len(dropped)} dropped {dropped}"
      + ("" if avail else " (query-metrics unavailable: list not pruned)"))
EOF
fi
METRICS=$(grep -vE '^\s*(#|$)' "$USED" | paste -sd,)

# Sections: verified identifiers; drop any this ncu does not list rather than fail the run.
SECTIONS=""
for s in SpeedOfLight SpeedOfLight_RooflineChart MemoryWorkloadAnalysis Occupancy LaunchStats WarpStateStats SchedulerStats ComputeWorkloadAnalysis; do
  if [ -s "$PROF/ncu_list_sections.txt" ] && ! grep -qw "$s" "$PROF/ncu_list_sections.txt"; then echo "section $s not listed by ncu; skipped"; continue; fi
  SECTIONS="$SECTIONS --section $s"
done

CLOCK=base; DEGRADED=0; NFAIL=0
for CFG in C64 C128; do
  [ "$CFG" = C64 ] && { CFGSTR=$C64; SUF=""; } || { CFGSTR=$C128; SUF="_$CFG"; }
  while IFS=$'\t' read -r SHORT REGEX; do
    [ "$DEGRADED" = 1 ] && break
    REP=$PROF/stage${STAGE}_${SHORT}${SUF}; LOG=$REP.ncu.log
    for attempt in 1 2; do
      echo "ncu stage=$STAGE kernel=$REGEX cfg=$CFGSTR dtype=$DTYPE clock-control=$CLOCK -> $REP.ncu-rep"
      set +e
      # shellcheck disable=SC2086
      ncu --kernel-name "regex:$REGEX" --launch-skip 10 --launch-count 1 --clock-control "$CLOCK" --cache-control all \
          $SECTIONS --metrics "$METRICS" -f -o "$REP" \
          "$BENCH" --stage "$STAGE" --cfg "$CFGSTR" --dtype "$DTYPE" --causal 0 --iters 12 > "$LOG" 2>&1
      rc=$?; set -e
      echo "ncu_cmd[${SHORT}${SUF}]=ncu --kernel-name regex:$REGEX --launch-skip 10 --launch-count 1 --clock-control $CLOCK --cache-control all$SECTIONS --metrics <ncu_metrics.used.txt> -f -o $REP $BENCH --stage $STAGE --cfg $CFGSTR --dtype $DTYPE --causal 0 --iters 12  (rc=$rc)" >> "$META"
      if grep -qiE 'ERR_NVGPUCTRPERM|does not have permission to access NVIDIA GPU Performance Counters' "$LOG"; then
        record "hardware counters BLOCKED on this runtime ($NCU_VER). Verbatim ncu output:"; cat "$LOG" >> "$PTXT"
        DEGRADED=1; break
      fi
      if [ $rc -ne 0 ] && [ "$CLOCK" = base ] && grep -qiE 'clock' "$LOG"; then
        record "ncu refused --clock-control base ($(grep -iE 'clock' "$LOG" | head -1)); retrying with none"
        CLOCK=none; echo "ncu_clock_control=none" >> "$OUT/env.txt"; continue
      fi
      break
    done
    [ "$DEGRADED" = 1 ] && break
    if [ $rc -ne 0 ]; then
      record "ncu FAILED rc=$rc for kernel=$REGEX cfg=$CFGSTR (see $LOG): $(grep -E '==ERROR==' "$LOG" | head -2 | tr '\n' ' ')"
      NFAIL=$((NFAIL + 1)); continue
    fi
    if ! grep -qE 'No kernels were profiled' "$LOG" && [ -f "$REP.ncu-rep" ]; then
      # `--print-units base` keeps bytes in bytes and rates per second; if this ncu refuses the
      # flag the plain import prints auto-scaled units, which fa_common.parse_ncu_raw_csv
      # normalises (or withholds). Which path was taken is auditable in env.txt and the .meta.
      if ncu --import "$REP.ncu-rep" --csv --page raw --print-units base > "$REP.raw.csv" 2> "$REP.import.log"; then
        UNITS=base
      else
        ncu --import "$REP.ncu-rep" --csv --page raw > "$REP.raw.csv" 2>> "$REP.import.log"; UNITS=auto
      fi
      grep -q "ncu_print_units=" "$OUT/env.txt" 2>/dev/null || echo "ncu_print_units=$UNITS" >> "$OUT/env.txt"
      echo "print_units[${SHORT}${SUF}]=$UNITS" >> "$META"
      echo "  -> $(wc -l < "$REP.raw.csv") lines in $REP.raw.csv (units: $UNITS)"; rm -f "$REP.import.log"
    else
      record "no launch matched regex:$REGEX for cfg=$CFGSTR (kernel not launched at this stage/cfg?)"; NFAIL=$((NFAIL + 1))
    fi
  done < <("$PY" scripts/kernel_regex.py "$STAGE")
  [ "$DEGRADED" = 1 ] && break
done
grep -q "ncu_clock_control=" "$OUT/env.txt" || echo "ncu_clock_control=$CLOCK" >> "$OUT/env.txt"
echo "clock_control=$CLOCK" >> "$META"; echo "degraded=$DEGRADED" >> "$META"
if [ "$DEGRADED" = 1 ]; then nsys_fallback; fi
echo "profile.sh done: degraded=$DEGRADED failures=$NFAIL raw_csv=$(ls "$PROF"/stage${STAGE}_*.raw.csv 2>/dev/null | wc -l | tr -d ' ')"
