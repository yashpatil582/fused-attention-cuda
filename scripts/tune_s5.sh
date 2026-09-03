#!/usr/bin/env bash
# S5 tuning sweep on the current GPU: benchmark every instantiated (BR, BC, PREFETCH) variant of
# fa_s5_tuned at the canonical configs and append the rows, tagged with the variant, to
# bench/results/<gpu>/stage5_tuning.csv. The winner per D becomes the S5 default in a separate
# commit (docs/STAGES.md keep rule: >= 3 % on the median with p10/p90 shown), never silently.
#
# usage: scripts/tune_s5.sh --gpu t4 [--iters 100]
# WHY a separate CSV: the README ladder rows (stage5.csv) must come from the committed defaults;
# the sweep is the evidence for choosing them.
set -euo pipefail
GPU=t4; ITERS=100
while [[ $# -gt 0 ]]; do case $1 in --gpu) GPU=$2; shift 2;; --iters) ITERS=$2; shift 2;; *) echo "unknown $1"; exit 2;; esac; done
PY=$(command -v python3 || command -v python)
OUT=bench/results/$GPU/stage5_tuning.csv; mkdir -p "bench/results/$GPU" build
LOCK=$(nvidia-smi --query-gpu=clocks.default_applications.graphics --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)
LOCKED=0; if [ -n "$LOCK" ] && nvidia-smi -lgc "$LOCK,$LOCK" >/dev/null 2>&1; then LOCKED=1; trap 'nvidia-smi -rgc >/dev/null 2>&1 || true' EXIT; fi
echo "tune_s5: gpu=$GPU iters=$ITERS clock_lock=${LOCKED}:${LOCK}MHz"
# (D, BR, BC, PREFETCH) — must match the instantiation table in src/stage5_tuned.cu.
VARIANTS_64="32:16 32:32 32:64 64:16 64:32 64:64 128:16 128:32"
VARIANTS_128="32:16 32:32 64:16"
run_variant() {  # cfg BR BC PF
  local cfg=$1 br=$2 bc=$3 pf=$4 tmp=build/tune_s5_tmp.csv
  rm -f "$tmp"
  if FA_FORCE_BR=$br FA_FORCE_BC=$bc FA_S5_PREFETCH=$pf "$PY" bench/bench.py --stage 5 --gpu "$GPU" --cfg "$cfg" \
       --baselines none --iters "$ITERS" --out "$tmp" ${LOCKED:+--clock-locked "$LOCK"} > build/tune_s5.log 2>&1; then
    "$PY" - "$tmp" "$OUT" "br${br}_bc${bc}_pf${pf}" <<'PYEOF'
import csv, os, sys
src, dst, tag = sys.argv[1:4]
rows = list(csv.DictReader(open(src)))
new = not os.path.exists(dst)
with open(dst, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()) if rows else None
    if w and new: w.writeheader()
    for r in rows:
        r["notes"] = (r["notes"] + ";" if r["notes"] else "") + "variant=" + tag
        w.writerow(r)
        print(f"TUNE_ROW variant={tag} cfg=B{r['B']}_H{r['H']}_N{r['N']}_D{r['D']} dtype={r['dtype']} ms_median={r['ms_median']} p10={r['ms_p10']} p90={r['ms_p90']} tflops={r['tflops']}")
PYEOF
  else
    echo "TUNE_ROW variant=br${br}_bc${bc}_pf${pf} cfg=$cfg FAILED ($(grep -m1 -iE 'error|invalid' build/tune_s5.log | cut -c1-120))"
  fi
}
for v in $VARIANTS_64; do for pf in 0 1; do run_variant C64 "${v%%:*}" "${v##*:}" "$pf"; done; done
for v in $VARIANTS_128; do for pf in 0 1; do run_variant C128 "${v%%:*}" "${v##*:}" "$pf"; done; done
echo "tune_s5: done -> $OUT"
"$PY" - "$OUT" <<'PYEOF'
import csv, sys, collections
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["ms_median"]]
best = {}
for r in rows:
    key = (r["D"], r["dtype"]); ms = float(r["ms_median"])
    if key not in best or ms < best[key][0]: best[key] = (ms, r["notes"].split("variant=")[-1], r["ms_p10"], r["ms_p90"])
for (d, dt), (ms, tag, p10, p90) in sorted(best.items()):
    print(f"BEST D={d} {dt}: {tag} median={ms:.4f} ms (p10 {p10}, p90 {p90})")
PYEOF
