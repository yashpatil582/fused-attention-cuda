#!/usr/bin/env bash
# Colab / VM entry point (judge_plan §4.6). Runs on the GPU box, never on the Mac.
#
#   usage: gpu_run.sh --stage N --gpu t4|a10|... --mode quick|full [--push] [--sweep] [--gpt]
#
#   --stage 0      smoke only: build + ctest + sanitizer + op smoke, no bench/profile
#   --mode quick   build + ctest + sanitizer + pip install -e + pytest -m gpu
#   --mode full    quick + bench.py + profile.sh (+ gpt_block.py with --gpt)
#   --sweep        bench.py runs the CONTRACT §5 sweep (48 configs, causal 0 and 1) instead of
#                  the two canonical configs
#   --push         commit bench/results + profiles as "results(<gpu>): stage N @ <sha>" and
#                  push with GH_PAT (outbound git is the only transport off Colab)
#
# Design: every step records its outcome; an EXIT trap prints a <=40-line SUMMARY block that
# is the LAST thing on the page, so the lead can read the outcome via get_page_text even when
# a step failed (the failing step is named). Idempotent: reruns reuse build/, append CSVs.
# Clocks: before the bench step the SM clock is locked to the driver's default application
# clock (`nvidia-smi -lgc`, verified to work as root on Colab, PROBE 2 2026-09-02) and reset
# on exit, so bench rows and the ncu profiles (`--clock-control base`) are at the same clock.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGE=1; GPU=t4; MODE=full; PUSH=0; SWEEP=0; GPT=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --stage) STAGE=$2; shift 2;;
    --gpu) GPU=$2; shift 2;;
    --mode) MODE=$2; shift 2;;
    --push) PUSH=1; shift;;
    --sweep) SWEEP=1; shift;;
    --gpt) GPT=1; shift;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
PY=$(command -v python3 || command -v python)
OUT=bench/results/$GPU; PROF=profiles/$GPU
# Everything this script prints is also kept in build/gpu_run_stage<N>.log, so a lost notebook
# cell (Colab reclaim, accidental delete) never loses the SUMMARY or the step logs.
mkdir -p build; exec > >(tee -a "build/gpu_run_stage${STAGE}.log") 2>&1
mkdir -p "$OUT" "$PROF" build
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
# `-dirty` when a TRACKED source file differs from HEAD (CONTRACT §3: commit = source under
# test). bench/results and profiles are excluded because this run itself appends to them.
if [ -n "$(git status --porcelain --untracked-files=no -- . ':(exclude)bench/results' ':(exclude)profiles' 2>/dev/null)" ]; then
  SHA="$SHA-dirty"
fi
export FA_COMMIT=$SHA   # fa_bench stamps its rows with this (bench_main.cu base_row)
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo HEAD)
JOBS=$(nproc 2>/dev/null || echo 2)
DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ---- summary state (filled in by the steps, printed by the trap) ---------------------
STEP=start; FAILED=""; BUILD=not-run; CTEST=not-run; SANITIZER=not-run; PYTEST=not-run
NCU_VERDICT=not-run; PUSH_STATUS=skipped; BENCH_STATUS=not-run; PROFILE_STATUS=not-run
LOCKED=0; LOCK_MHZ=""; CLOCK_STATUS="not attempted"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)

summary() {
  # The trap inherits `set -euo pipefail`; nothing in here may abort the summary itself.
  set +e; set +o pipefail
  echo
  echo "====== FA SUMMARY ======"
  echo "gpu: $GPU ($GPU_NAME)  sha: $SHA  branch: $BRANCH  stage: $STAGE  mode: $MODE"
  echo "build: $BUILD"
  if [ -s "$OUT/stage${STAGE}_ptxas.txt" ]; then
    "$PY" - "$OUT/stage${STAGE}_ptxas.txt" <<'PYEOF' | head -8
import re, sys
name, spill, lines = None, "spill=n/a", []
for line in open(sys.argv[1]):
    m = re.search(r"Compiling entry function '([^']+)' for '(sm_\d+)'", line)
    if m:
        k = re.search(r"fa_s\d_[a-z_]+|fa_smoke", m.group(1)); name = (k.group(0) if k else m.group(1)[:40]) + "@" + m.group(2)
        spill = "spill=n/a"
    if "spill stores" in line:
        s = re.findall(r"(\d+) bytes spill (stores|loads)", line); spill = " ".join(f"spill_{t}={n}" for n, t in s)
    # nvcc >= 12.4 prints `Used N registers, used K barriers, M bytes smem, ...`: skip the barriers clause
    m = re.search(r"Used (\d+) registers(?:.*?, (\d+) bytes smem)?", line)
    if m and name:
        lines.append(f"  {name}: regs={m.group(1)} smem={m.group(2) or 0} {spill}"); name = None
print("ptxas:" if lines else "ptxas: (no kernel lines in report)"); print("\n".join(lines))
PYEOF
  else
    echo "ptxas: no report (nothing recompiled this run or build failed)"
  fi
  echo "ctest: $CTEST"
  echo "sanitizer: $SANITIZER"
  echo "pytest(gpu): $PYTEST"
  echo "clocks: $CLOCK_STATUS"
  echo "bench: $BENCH_STATUS"
  if [ -f build/bench.log ]; then grep -E '^BENCH_ROW' build/bench.log | grep -E 'impl=(naive|tiled|fused|wmma|tuned|sdpa_efficient) ' | sed -E 's/^BENCH_ROW /  /; s/ p10=[^ ]* p90=[^ ]*//' | head -12; fi
  if [ -f build/gpt.log ]; then grep -E '^GPT_ROW' build/gpt.log | sed 's/^GPT_ROW /  gpt: /' | head -3; fi
  echo "ncu: $NCU_VERDICT; profiles: $(ls "$PROF"/stage${STAGE}_*.raw.csv 2>/dev/null | wc -l | tr -d ' ') raw csv; $PROFILE_STATUS"
  [ -f "$PROF/stage${STAGE}.profiler.txt" ] && echo "profiler.txt: $(head -1 "$PROF/stage${STAGE}.profiler.txt" | cut -c1-120)"
  echo "push: $PUSH_STATUS"
  [ -n "$FAILED" ] && echo "FAILED STEP: $FAILED"
  echo "====== END SUMMARY ======"
}
on_exit() {
  rc=$?
  if [ $rc -ne 0 ] && [ -z "$FAILED" ]; then FAILED="$STEP (rc=$rc)"; fi
  if [ "$LOCKED" = 1 ]; then nvidia-smi -rgc >/dev/null 2>&1 || true; fi   # never leave the GPU locked
  summary
}
trap on_exit EXIT
step() { STEP=$1; echo; echo "### [$STEP] $(date -u +%H:%M:%S)"; }

# ---- 1. environment provenance -------------------------------------------------------
step env
{
  echo "commit=$SHA"; echo "branch=$BRANCH"; echo "date_utc=$DATE_UTC"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,clocks.max.sm,clocks.sm,clocks.default_applications.graphics --format=csv 2>&1 || echo "nvidia-smi=missing"
  nvcc --version 2>/dev/null | tail -1 || echo "nvcc=missing"
  "$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.get_arch_list())" 2>&1 || echo "torch=missing"
  (ncu --version 2>/dev/null | head -1) || echo "ncu=missing"
  (nsys --version 2>/dev/null | head -1 | sed 's/^/nsys /') || echo "nsys=missing"
  (cmake --version 2>/dev/null | head -1) || echo "cmake=missing"
  echo "compute-sanitizer=$(command -v compute-sanitizer || ls /usr/local/cuda/bin/compute-sanitizer 2>/dev/null || echo missing)"
} > "$OUT/env.txt"
cat "$OUT/env.txt"
# CMakePresets.json is schema v6: `cmake --preset` needs cmake >= 3.25 (Ubuntu 22.04 apt ships
# 3.22), so check the VERSION, not just the binary; ninja is the presets' generator.
cmake_ok() { cmake --version 2>/dev/null | awk 'NR==1{split($3,v,"."); exit !(v[1]>3 || (v[1]==3 && v[2]>=25))}'; }
cmake_ok || "$PY" -m pip install -q 'cmake>=3.25' ninja
command -v ninja >/dev/null || "$PY" -m pip install -q ninja
"$PY" -c "import yaml, pytest" 2>/dev/null || "$PY" -m pip install -q pyyaml pytest
# Default dtype of the stage (bench/configs.yaml stage_dtype): S4/S5 kernels reject fp32, so every
# fa_bench invocation below must carry it (profile.sh already does).
DTYPE=$("$PY" scripts/kernel_regex.py --dtype "$STAGE" 2>/dev/null || echo fp32)

# ---- 2. configure + build (preset colab/vm by --gpu; explicit arch fallback) -----------
step configure
PRESET=$([ "$GPU" = t4 ] && echo colab || echo vm)
ARCH=$([ "$GPU" = t4 ] && echo 75-real || echo native)
if [ -f CMakePresets.json ] && grep -q "\"name\": *\"$PRESET\"" CMakePresets.json && cmake_ok; then
  cmake --preset "$PRESET" -B build > build/configure.log 2>&1 || { tail -30 build/configure.log; BUILD=FAIL; exit 1; }
else
  echo "preset $PRESET unusable (missing from CMakePresets.json or cmake < 3.25); configuring explicitly (arch=$ARCH)"
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH" > build/configure.log 2>&1 || { tail -30 build/configure.log; BUILD=FAIL; exit 1; }
fi
step build
if cmake --build build -j"$JOBS" > build/build.log 2>&1; then BUILD=ok; else BUILD=FAIL; tail -40 build/build.log; exit 1; fi
# ptxas prints the spill line indented with no `ptxas` prefix (same pattern as build.yml).
grep -E '^ptxas (info|warning)|bytes spill|bytes stack frame' build/build.log > build/ptxas.tmp || true
# ninja only recompiles what changed: keep the previous report instead of overwriting it with nothing
if [ -s build/ptxas.tmp ]; then mv build/ptxas.tmp "$OUT/stage${STAGE}_ptxas.txt"; else rm -f build/ptxas.tmp; echo "no ptxas output (nothing recompiled)"; fi
grep -E 'error|warning' build/build.log | head -20 || true
BENCH=$(ls build/bench/fa_bench build/fa_bench 2>/dev/null | head -1 || true)

# ---- 3. ctest ------------------------------------------------------------------------
step ctest
set +e; ctest --test-dir build --output-on-failure > build/ctest.log 2>&1; rc=$?; set -e
tail -5 build/ctest.log
if [ $rc -eq 0 ]; then CTEST=$(grep -E 'tests passed' build/ctest.log | tail -1 || echo ok)
elif grep -q 'No tests were found' build/ctest.log; then CTEST="no tests registered"
else CTEST="FAIL ($(grep -E 'tests passed' build/ctest.log | tail -1))"; exit $rc; fi

# ---- 4. compute-sanitizer gate (memcheck + racecheck on fa_bench --smoke) -------------
step sanitizer
CS=$(command -v compute-sanitizer || ls /usr/local/cuda/bin/compute-sanitizer 2>/dev/null || true)
if [ -z "$CS" ]; then SANITIZER="missing (recorded in env.txt)"; echo "sanitizer=missing" >> "$OUT/env.txt"
elif [ -z "$BENCH" ]; then SANITIZER="fa_bench not built"
else
  SANITIZER=""
  : > "$OUT/stage${STAGE}_sanitizer.txt"   # one run per file: reruns replace, never accumulate
  # Small shapes on purpose: memcheck instruments every global access, so the naive kernels at
  # C64 (67M threads x D loads) take many minutes under it; N=288 (= 9*32, not a multiple of 64)
  # still exercises multi-tile loops, a partial last tile for BC=64, and both head dims.
  SAN_CFGS="B1_H2_N288_D64 B1_H2_N288_D128"
  for tool in memcheck racecheck; do
    # --error-exitcode: both tools exit 9 on any reported error, independent of the summary
    # wording (memcheck: `ERROR SUMMARY: N errors`; racecheck: `RACECHECK SUMMARY: N hazards
    # displayed (E errors, W warnings)`). Both formats are parsed for the summary line only.
    : > "build/sanitizer_$tool.log"; rc=0
    for cfg in $SAN_CFGS; do
      set +e; "$CS" --tool $tool --error-exitcode 9 "$BENCH" --smoke --stage "$STAGE" --dtype "$DTYPE" --cfg "$cfg" >> "build/sanitizer_$tool.log" 2>&1; r=$?; set -e
      [ $r -ne 0 ] && rc=$r
    done
    errs=$(grep -oE 'ERROR SUMMARY: [0-9]+ error|RACECHECK SUMMARY: [0-9]+ hazards? displayed \([0-9]+ error' "build/sanitizer_$tool.log" | grep -oE '[0-9]+ error' | grep -oE '[0-9]+' | awk '{s+=$1} END {print s+0}' || true)
    errs=${errs:-?}
    SANITIZER="$SANITIZER$tool=$errs errors(rc=$rc) "
    { echo "# compute-sanitizer --tool $tool  commit=$SHA  date=$DATE_UTC  stage=$STAGE  dtype=$DTYPE  rc=$rc"; cat "build/sanitizer_$tool.log"; echo; } >> "$OUT/stage${STAGE}_sanitizer.txt"
    if [ $rc -ne 0 ] || { [ "$errs" != "?" ] && [ "$errs" != "0" ]; }; then tail -20 "build/sanitizer_$tool.log"; exit 1; fi
  done
fi

# ---- 5. ncu go/no-go on the real binary (one metric, one launch) ---------------------
step ncu-verdict
if ! command -v ncu >/dev/null; then NCU_VERDICT="missing"
elif [ -z "$BENCH" ]; then NCU_VERDICT="fa_bench not built"
else
  set +e; ncu --launch-count 1 --metrics sm__cycles_elapsed.avg "$BENCH" --smoke --stage "$STAGE" --dtype "$DTYPE" > "$OUT/ncu_probe.txt" 2>&1; set -e
  if grep -qiE 'ERR_NVGPUCTRPERM|does not have permission' "$OUT/ncu_probe.txt"; then NCU_VERDICT="BLOCKED (ERR_NVGPUCTRPERM)"
  elif grep -qE 'sm__cycles_elapsed\.avg +[a-z]* +[0-9]' "$OUT/ncu_probe.txt"; then NCU_VERDICT="PASS (counters readable)"
  else NCU_VERDICT="UNKNOWN (see $OUT/ncu_probe.txt)"; fi
fi
echo "ncu_verdict=$NCU_VERDICT" >> "$OUT/env.txt"; echo "$NCU_VERDICT"

# ---- 6. torch extension + gpu pytest -------------------------------------------------
step pip-install
ARCH_LIST=$([ "$GPU" = t4 ] && echo 7.5 || "$PY" -c "import torch;print('%d.%d'%torch.cuda.get_device_capability())")
TORCH_CUDA_ARCH_LIST="$ARCH_LIST" MAX_JOBS="$JOBS" "$PY" -m pip install -q --no-build-isolation -e python/ > build/pip.log 2>&1 || { tail -30 build/pip.log; exit 1; }
"$PY" -c "import fused_attn, torch; print('op:', torch.ops.fused_attn.forward)"
step pytest-gpu
set +e; "$PY" -m pytest -q tests -m gpu --stage "$STAGE" > build/pytest.log 2>&1; rc=$?; set -e
tail -15 build/pytest.log
PYTEST=$(grep -E '^(=+ )?[0-9]+ (passed|failed|skipped|error)' build/pytest.log | tail -1 | sed -E 's/=+//g; s/ in [0-9.]+s//' || true)
if [ $rc -eq 5 ]; then PYTEST="no gpu tests collected"; elif [ $rc -ne 0 ]; then PYTEST="FAIL: ${PYTEST:-see build/pytest.log}"; exit $rc; fi

# ---- 7. bench + profile (full mode, stage >= 1) --------------------------------------
lock_clocks() {
  # BENCHMARK_PROTOCOL rule 4: lock the SM clock to the driver's default application clock
  # (the base clock ncu --clock-control base uses; 585 MHz on T4), read from the driver, never
  # typed. Refusal (no root, unsupported) is recorded and the rows say clock=unlocked.
  LOCK_MHZ=$(nvidia-smi --query-gpu=clocks.default_applications.graphics --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if ! [[ "$LOCK_MHZ" =~ ^[0-9]+$ ]]; then
    LOCK_MHZ=$(nvidia-smi -q -d CLOCK 2>/dev/null | awk '/Default Applications Clocks/{f=1;next} f&&/Graphics/{print $3; exit}')
  fi
  if [[ "$LOCK_MHZ" =~ ^[0-9]+$ ]] && nvidia-smi -lgc "$LOCK_MHZ,$LOCK_MHZ" >/dev/null 2>&1; then
    LOCKED=1; CLOCK_STATUS="locked to $LOCK_MHZ MHz (nvidia-smi -lgc; -rgc on exit)"
    echo "clock_lock=$LOCK_MHZ" >> "$OUT/env.txt"
  else
    CLOCK_STATUS="unlocked (nvidia-smi -lgc refused or default application clock unreadable: '${LOCK_MHZ:-?}')"
    echo "clock_lock=refused" >> "$OUT/env.txt"
  fi
  echo "$CLOCK_STATUS"
}
if [ "$STAGE" = 0 ]; then
  BENCH_STATUS="skipped (stage 0 = smoke only)"; PROFILE_STATUS="skipped"
elif [ "$MODE" = full ]; then
  step clock-lock
  lock_clocks
  step bench
  BENCH_ARGS=(--stage "$STAGE" --gpu "$GPU" --out "$OUT/stage${STAGE}.csv")
  [ "$SWEEP" = 1 ] && BENCH_ARGS+=(--sweep --causal both)   # CONTRACT §5: the sweep includes causal 0 and 1
  [ "$LOCKED" = 1 ] && BENCH_ARGS+=(--clock-locked "$LOCK_MHZ")   # bench.py verifies the lock per row
  set +e; "$PY" bench/bench.py "${BENCH_ARGS[@]}" > build/bench.log 2>&1; rc=$?; set -e
  grep -E '^(BENCH_ROW|\[bench\])' build/bench.log | head -60
  BENCH_STATUS="$(grep -c '^BENCH_ROW' build/bench.log || true) rows -> $OUT/stage${STAGE}.csv (rc=$rc)"
  [ $rc -ne 0 ] && { tail -20 build/bench.log; exit $rc; }
  if [ "$GPT" = 1 ]; then
    step gpt-block
    GPT_ARGS=(--gpu "$GPU"); [ "$LOCKED" = 1 ] && GPT_ARGS+=(--clock-locked "$LOCK_MHZ")
    "$PY" bench/gpt_block.py "${GPT_ARGS[@]}" > build/gpt.log 2>&1 || { tail -20 build/gpt.log; exit 1; }
    grep -E '^GPT_ROW|\[gpt_block\]' build/gpt.log
  fi
  step profile
  if bash scripts/profile.sh "$STAGE" "$GPU" > build/profile.log 2>&1; then PROFILE_STATUS="profile.sh ok"; else PROFILE_STATUS="profile.sh FAILED (see build/profile.log)"; tail -30 build/profile.log; exit 1; fi
  tail -15 build/profile.log
else
  BENCH_STATUS="skipped (mode=$MODE)"; PROFILE_STATUS="skipped"
fi

# ---- 8. push results (outbound git only; PAT never echoed) ---------------------------
if [ "$PUSH" = 1 ]; then
  step push
  if [ -z "${GH_PAT:-}" ]; then PUSH_STATUS="skipped (GH_PAT unset)"
  elif [ "$BRANCH" = HEAD ]; then PUSH_STATUS="refused (detached HEAD)"
  else
    big=$(find bench/results profiles -type f -size +20M 2>/dev/null || true)
    if [ -n "$big" ]; then echo "artifact >20MB, refusing to commit:"; echo "$big"; PUSH_STATUS="refused (file >20MB)"; exit 3; fi
    git add bench/results profiles
    if git -c user.name=colab-runner -c user.email=colab@local commit -q -m "results($GPU): stage $STAGE @ $SHA"; then
      if git push -q "https://x-access-token:${GH_PAT}@github.com/yashpatil582/fused-attention-cuda.git" "HEAD:refs/heads/$BRANCH" > build/push.log 2>&1; then
        PUSH_STATUS="ok $(git rev-parse --short HEAD) -> $BRANCH"
      else sed "s/${GH_PAT}/***/g" build/push.log | tail -5; PUSH_STATUS="FAILED (see build/push.log)"; exit 1; fi
    else PUSH_STATUS="nothing new to commit"; fi
  fi
fi
STEP=done
