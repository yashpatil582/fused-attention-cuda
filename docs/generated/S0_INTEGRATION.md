# S0 integration record (Mac, no GPU) — 2026-09-02

Evidence of what was run on the integration machine after the six parallel S0 builders finished,
what passed, what was changed to make the tree consistent, and what could NOT be verified here.
Nothing below ran on a GPU. Every number is copied from a command's output on this machine.

Machine: macOS (Darwin 25.4.0, arm64 M2), `/opt/anaconda3/bin/python3` 3.12.7, torch 2.5.1 (CPU
wheel), numpy 1.26.4, pandas 2.2.3, pyyaml 6.0.2, matplotlib 3.9.2. Installed for this pass:
cmake 4.4.3 + ninja 1.13.2 (`brew install cmake ninja`); ruff 0.15.4, clang-format 23.1.0, gersemi
0.28.1, pre-commit 4.6.2, check-jsonschema (pip, `/opt/anaconda3`). Docker: colima daemon up,
`nvidia/cuda:12.8.1-devel-ubuntu22.04` (linux/arm64) -> `nvcc V12.8.93`; the repo's
`docker/Dockerfile.nvcc` image `fa-nvcc:12.8.1` was built by `scripts/local_compile.sh`.

## 1. Cross-owner consistency (all consistent after the edits in section 3)

Checked programmatically (`python -` snippet over the files; every line printed `True`):

| What | Where it must agree | Result |
|---|---|---|
| CSV schema (21 columns) | `CONTRACT.md` §3 = `scripts/fa_common.py::SCHEMA` = `bench/bench_main.cu::kCsvHeader` | identical |
| `impl` vocabulary | `CONTRACT.md` §3 = `fa_common.IMPL_ORDER` | identical (`smoke`/`gpt_*` accepted by the sanity test only, see §3) |
| kernel names | `CONTRACT.md` §2 = `scripts/kernel_regex.py::KERNELS`; names in `src/dispatch.cu` comments ⊆ contract; `fa_smoke` defined at global scope in `src/smoke.cu` | identical |
| README generated-block markers | `CONTRACT.md` §6 = `fa_common.BLOCK_TAGS` = markers present in `README.md` | identical |
| op schemas | `CONTRACT.md` §4 = `python/fused_attn/ext.cpp` `TORCH_LIBRARY` = `ops.py` `register_fake` names; call sites `bench/bench.py` (`forward_stage(q,k,v,causal,scale,stage)[0]`), `bench/gpt_block.py` (`forward(q,k,v,True,scale)`) | identical |
| `sources.txt` parsing | CMake `file(STRINGS ... REGEX "^[^#].*\.cu[ \t]*$")` vs `python/setup.py::kernel_sources()` | both yield `[src/dispatch.cu, src/smoke.cu]` (`setup.kernel_sources()` printed `['../src/dispatch.cu', '../src/smoke.cu']`; CMake printed `sources=src/dispatch.cu;src/smoke.cu`) |
| canonical configs | `CONTRACT.md` §5 = `bench/configs.yaml` (`C64` = B2 H8 N2048 D64, `C128` = D128) = hard-coded aliases in `bench_main.cu::parse_cfg` | identical |
| CMake presets referenced | `gpu_run.sh` (`colab`,`vm`), `Makefile`/`build.yml` (`cpu-only`,`ci`), `local_compile.sh` (`docker`) ⊆ `CMakePresets.json` | all defined |
| `fa_bench` path | `gpu_run.sh`/`profile.sh` expect `build/bench/fa_bench` (fallback `build/fa_bench`); CMake `RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bench`; `docker` preset build produced `build/docker/bench/fa_bench` | consistent |
| `cmake --preset colab -B build` (gpu_run.sh) | `-B` must override the preset's `binaryDir` | verified with cmake 4.4.3: `cmake --preset cpu-only -B /tmp/fa_presetB` wrote `/tmp/fa_presetB/CMakeCache.txt` |
| `pytest --stage N` (gpu_run.sh) | option must exist in `tests/conftest.py` | exists; `pytest -q tests --stage 3 -m gpu` -> `77 skipped, 347 deselected` |
| GitHub `ubuntu-24.04` runner tools (build.yml calls `ninja --version`; presets use the Ninja generator) | runner image README (fetched 2026-09-02) | lists CMake 3.31.6 and Ninja 1.13.2 -> no CI edit needed |

## 2. What ran and passed

| Step | Command (from repo root) | Result |
|---|---|---|
| ruff | `python -m ruff check .` / `python -m ruff format --check .` | `All checks passed!` / `20 files already formatted` (ruff 0.15.4; CI pins 0.16.5 — not installed here, drift possible but unlikely for these rules) |
| clang-format | `find ... -name '*.cu' -o '*.cuh' -o '*.cpp' -o '*.h' \| xargs clang-format --dry-run --Werror --style=file --fallback-style=none` (12 files) | clean, rc=0 (clang-format 23.1.0 = the CI/pre-commit pin) |
| gersemi | `gersemi --check CMakeLists.txt` | clean |
| workflow schemas | `check-jsonschema --builtin-schema vendor.github-workflows build.yml lint.yml`; `vendor.dependabot dependabot.yml` | `ok -- validation done` (x2) |
| pre-commit (CI parity) | `pre-commit run --files $(git ls-files --others --exclude-standard)` | all 8 hooks Passed: trailing-whitespace, end-of-file-fixer, check-yaml, check-merge-conflict, clang-format v23.1.0, ruff-check v0.16.5, ruff-format v0.16.5, gersemi 0.28.1. Note: the hook envs pin ruff 0.16.5, so ruff 0.16.5 DID run clean here via pre-commit. |
| CPU reference (CMake) | `cmake --preset cpu-only && cmake --build build/cpu-only && ctest --test-dir build/cpu-only --output-on-failure` | configure OK (AppleClang, `FA_ENABLE_CUDA=OFF`), build 4/4 targets, `1/1 Test #1: test_cpu_reference ... Passed 0.25 sec`, `100% tests passed`. `cmake --build --preset cpu-only` / `ctest --preset cpu-only` (the `make cpu-test` form) also work. |
| CPU pytest | `python -m pytest -q -m "not gpu"` | `153 passed, 2 skipped, 269 deselected` (skips = `test_bench_sanity` with no CSV yet). GPU files alone: `269 skipped` (no CUDA device). |
| docs generators | `python scripts/bench_to_md.py && python scripts/ncu_to_md.py && python scripts/roofline.py` | `bench_to_md: no change`; `roofline: no profiles/<gpu>/*.raw.csv ...; nothing to plot`; README byte-identical before/after; then `--check` for both: `up to date`; `pytest tests/test_docs_cpu.py` -> `5 passed` |
| notebook / YAML / JSON / bash | `json.load(colab/runner.ipynb)` (nbformat 4.5, 3 code cells); `yaml.safe_load` on `build.yml`, `lint.yml`, `.pre-commit-config.yaml`, `dependabot.yml`, `bench/configs.yaml`; `json.tool CMakePresets.json`; `bash -n` on the 6 shell scripts | all OK |
| scripts smoke | `kernel_regex.py 0/1/--cfg C64/--cfg C128/--dtype 1/--dtype 4`; `perf_trailer.py 3` (prints "unavailable" lines, no invented values); `bench.py --help`, `gpt_block.py --help`; `bench.py --gpu t4 --stage 1` -> `CUDA not available` rc=2; `bench.correctness_gate` CPU smoke -> `(True, 'max_err=6.436e-07 ...')`; `fused_attn.is_available()` -> `False` and `forward()` raises the build-hint `ImportError`; `.github/scripts/ptxas_summary.py` on synthetic and on the real docker report | all as designed |
| nvcc compile, sm_75 | `FA_ARCHS=75-real bash scripts/local_compile.sh` (docker, nvcc 12.8.93, arm64, cmake from PyPI inside the image) | configure + build OK: `src/smoke.cu`, `src/dispatch.cu`, `bench/bench_main.cu` compiled and `bench/fa_bench` linked; 19 ptxas lines; `fa_smoke` 10 regs, 0 spills; `fill_uniform<float/__half/__nv_bfloat16>` 10/10/14 regs, 0 spills; no compiler warnings in `build.log` (only the image's "NVIDIA Driver was not detected" banner) |
| nvcc compile, CI arch list | `bash scripts/local_compile.sh` (default `75-real;80-real;86-real;89-real;90-real;90-virtual`) | rc=0; 95 ptxas lines; `ptxas_summary.py`: **20 (kernel, arch) entries, 0 with register spills, max registers 14** (4 kernels x sm_75/80/86/89/90). This is the same arch list `cmake --preset ci` uses in `build.yml`, with the same nvcc major (12.8), on arm64 instead of x86-64. |
| `experiments/wmma_gemm/wmma_gemm.cu` | docker `nvcc -O3 -std=c++17 -lineinfo -arch=sm_75 -Xptxas -v -Xcompiler -Wall,-Wextra` then `/tmp/wmma_gemm 128 128 64 --ref-only` | no warnings; `fa_wmma_f16_kernel` 80 regs / 0 spills / 1 barrier, `fa_wmma_bf16_kernel` 4 regs (trap stub on sm_75); `--ref-only` printed `max|S_ref| 11.5229 mean|S_ref| 2.1182 mean|O_ref| 0.0644 err_ref 6.006e-05` = the values `check_vs_torch.py` documents |
| `python/fused_attn/ext.cpp` (host C++ binding) | in the `fa-nvcc:12.8.1` container: `g++ 11.4 -fsyntax-only -std=c++17 -Wall -Wextra -Wno-unused-parameter -DTORCH_EXTENSION_NAME=_C` with the REAL CUDA 12.8.1 headers (`/usr/local/cuda/include`), the REAL torch 2.5.1 headers (copied from the Mac's wheel, CPU build) + `python3-dev` 3.10 headers | rc=0, zero diagnostics. Caveat: the CPU wheel lacks the generated `c10/cuda/impl/cuda_cmake_macros.h`; a 1-macro stub (`#define C10_CUDA_BUILD_SHARED_LIBS`) was supplied for it. Everything else was genuine. On the Mac itself the same check stops at `cuda_runtime_api.h` (no CUDA headers), as expected. |

## 3. Edits made during integration (every one; diffs vs the pre-integration snapshot)

1. `include/fa/attention.cuh` — **frozen header, whitespace only**, `clang-format -i` (Google style,
   100 cols): the `forward(...)` parameter list repacked from 3 lines to 2, and one extra space before
   the `impl_name` trailing comment. No token changed. Required because `lint.yml` and the
   pre-commit hook format every `.cuh`; the alternative (exclude the file in both) was not taken.
   This is the only TRACKED file modified (`git diff --stat`: 3 insertions, 4 deletions).
2. `python/fused_attn/ext.cpp` — `clang-format -i` (6 whitespace re-wraps). clang-format's include
   regrouping then moved `#include <Python.h>` below the ATen headers; restored it to first position
   (CPython requires it) by adding a comment line after it that ends the include block, so the file
   is both Python.h-first and clang-format-clean.
3. `pyproject.toml` — `[tool.ruff] extend-exclude` gains `"*.ipynb"` (+ 2 comment lines). ruff lints
   notebooks by default; `colab/runner.ipynb` had 21 findings (E501 on `!` shell-magic lines that
   cannot be wrapped, one I001) and `ruff format --check` wanted to rewrite it, so `lint.yml` would
   have been red.
4. `scripts/gpu_run.sh` line 155 — `pip install -q -e python/` -> `pip install -q --no-build-isolation
   -e python/`. Without it pip builds in an isolated env where `torch` is not importable and
   `python/setup.py` (which imports `torch.utils.cpp_extension` at module level) fails on Colab;
   `setup.py`'s own docstring prescribes `--no-build-isolation`. Unverified on Colab (no Colab here).
5. `tests/test_bench_sanity.py` — the unknown-impl assertion also accepts `impl == "smoke"` (the S0
   `fa_bench --smoke` round-trip row, `impl=smoke,stage=0`), so a committed smoke CSV cannot fail CI;
   `IMPL_ORDER` (README rendering order) is unchanged, so a smoke row never enters the README
   headline table. Re-wrapped by `ruff format`.
6. `docs/STAGES.md` S0 entry — "CMake >= 3.24" -> "CMake >= 3.24 (>= 3.25 to use `CMakePresets.json`
   v6)", aligning the docs with the Makefile comment and the presets file.

Not edited, on purpose: `CONTRACT.md`; `.github/workflows/*` (runner already has ninja);
`bench/bench_main.cu` (745 lines) and `bench/bench.py` (500 lines) although both exceed the ~400-line
guideline; the `TBD` placeholders in `docs/ENV.md`, `docs/STAGES.md` probe table, `docs/RESEARCH.md`
§8 and `experiments/wmma_gemm/README.md` (they are the honest state).

## 4. What could NOT be verified on the Mac, and why

- **Anything on a GPU**: `fa_bench --smoke` (ctest `test_smoke`), `compute-sanitizer`, the `ncu`
  go/no-go and counters, `profile.sh`, `bench.py` rows, `gpt_block.py`, the `gpu_run.sh` summary
  block, `sync.sh`/`gate_gpu.sh` (need `ssh gpu`). No NVIDIA device exists here; the docker image
  has nvcc only (`NVIDIA Driver was not detected`). All probes L/1-7 remain TBD except PROBE L, which
  this record answers: arm64 nvcc-in-docker compiles the tree in seconds.
- **The torch extension build and link** (`nvcc` + torch CUDA wheel, `TORCH_LIBRARY` registration at
  import, `opcheck`, `torch.compile` graph-break test, the `--no-build-isolation` fix): the Mac's
  torch is a CPU wheel and the container has no torch. Only `ext.cpp` syntax against real headers was
  proven (section 2, with the one-macro stub caveat).
- **CI itself**: `build.yml`/`lint.yml` have not run (repo not pushed). Schema validation, the
  `pre-commit` parity run and the local compile with the exact `ci` arch list are the proxies. PROBE 6
  (nvcc-only apt install time) is unanswered. The CUDA 13.0.1 canary leg was not exercised (only the
  12.8.1 image is pulled locally).
- **x86-64 vs arm64**: every nvcc compile here was the linux/arm64 CUDA 12.8 toolchain (g++ 11.4).
  Colab/CI are x86-64; host-compiler differences (e.g. glibc/gcc 13 on ubuntu-24.04) are unverified.
- **ruff version drift**: direct runs used ruff 0.15.4; CI pins 0.16.5. The pre-commit hook run
  (which installs ruff 0.16.5) also passed, so the drift is covered for the current tree.
- **`nvidia-smi -lgc`, `--clock-control base`, `--print-units base`, the two UNVERIFIED ncu SOL
  metric IDs, nsys fallback**: all remain as documented in `docs/PROFILING_PROTOCOL.md` — PROBE 2.
- **`docs/ENV.md` roofs**: still `TBD` (parse to `{}`); `test_bench_sanity` applies no TFLOP/s cap
  until the lead pastes the T4 datasheet values + URL.

## 5. Residual items for owners (not blockers)

- scripts: `bench/bench.py` sets `torch.backends.cuda.matmul.allow_tf32 = False`; the protocol text
  says torch 2.14 deprecates that in favour of `fp32_precision = "ieee"`. Harmless warning at worst;
  align once the Colab torch version is known.
- build: `bench/bench_main.cu` writes `impl=smoke` rows only when `--out` is given; `gpu_run.sh`
  never passes `--out` to the smoke run, so the "round trip" commit will contain `env.txt`,
  `stage0_ptxas.txt`, `stage0_sanitizer.txt`, `ncu_probe.txt` but no CSV unless the lead runs
  `./build/bench/fa_bench --smoke --out bench/results/t4/stage0.csv --commit <sha>` by hand.
- ci: `Makefile`'s `lint` target and `lint.yml` both use `git ls-files`, so clang-format covers
  nothing until the S0 files are committed; this pass used `find` instead.
- docs: the frozen-header whitespace change (edit 1) should be acknowledged in the S0 commit message
  or reverted together with a clang-format exclusion for `include/fa/attention.cuh`.

## 6. Review findings applied (fixer pass, 2026-09-02)

Every finding of the S0 review was verified against the code before it was touched; the one that
was wrong is marked as skipped with the reason. After the edits the integration checks of section 2
were re-run: `ruff check` / `ruff format --check` clean (21 files); `clang-format --dry-run --Werror`
clean on every `.cu/.cuh/.cpp/.h`; `gersemi --check` clean; `pre-commit run --files <all S0 files>`
all hooks passed (pinned ruff 0.16.5 / clang-format 23.1.0); `cmake --preset cpu-only` + build +
`ctest` `100% tests passed`; `pytest -q -m "not gpu"` **175 passed, 3 skipped** (was 153: 22 new
CPU tests in `tests/test_fa_common_cpu.py`); `bench_to_md.py --check` / `ncu_to_md.py --check` up to
date; `roofline.py` "nothing to plot"; YAML/JSON/notebook loads and `bash -n` on every script OK;
`FA_ARCHS=75-real scripts/local_compile.sh` (docker, nvcc 12.8.93) rc=0 with the patched
`bench_main.cu`, zero compiler warnings, and the spill report now fires (`4 x "0 bytes spill"`).
Nothing ran on a GPU; every GPU-side behaviour below is still unverified on Colab.

| # | Finding (file:line) | Severity | Action |
|---|---|---|---|
| 1 | `gpu_run.sh:133` racecheck prints `RACECHECK SUMMARY: N hazards displayed (E errors, W warnings)`, no `ERROR SUMMARY`; gate exited 1 on every stage-0 run | blocker | Confirmed. Both tools now run with `--error-exitcode 9` and the gate is on `rc`; both summary formats are parsed for the summary line (`errs=?` when neither matches does not fail on its own). Parser checked against the four real summary strings. |
| 2 | `gpu_run.sh:132/145` `fa_bench --smoke --stage N` without `--dtype`; Options default fp32, Wmma/Tuned reject it (`dispatch.cu::contract_allows`) | major | Confirmed. `DTYPE=$(kernel_regex.py --dtype $STAGE)` computed once after the pyyaml install and passed to the sanitizer and the ncu probe. |
| 3 | `gpu_run.sh:110` / `local_compile.sh:58` `grep ^ptxas` drops the indented spill line (verified in `build/docker/build.log`) | major | Confirmed. Both use build.yml's pattern `'^ptxas (info\|warning)\|bytes spill\|bytes stack frame'`; `fa_common.parse_ptxas` and `ptxas_summary.py` report spills as `None`/`n/a` when no spill line was seen, never 0; `ncu_to_md` renders `n/a`. CPU test pins the real nvcc 12.8 format. |
| 4 | `fa_common.py:346` and two siblings: `Used N registers, used K barriers, M bytes smem` not matched (verified in build.log) | major | Confirmed. Regex is `Used (\d+) registers(?:.*?, (\d+) bytes smem)?` in `fa_common.PTXAS_USED_RE`, `ptxas_summary.py::_USED` and the gpu_run.sh SUMMARY heredoc. |
| 5 | `ops.py:90` opcheck default `test_autograd_registration` raises "no inputs have requires_grad=True" (verified in torch 2.5.1 `optests/autograd_registration.py`) | major | Confirmed. `opcheck_all` now runs `test_schema/test_faketensor/test_aot_dispatch_dynamic` on the plain inputs and `test_autograd_registration` on detached `requires_grad_()` copies, merging the dicts. GPU-unverified. |
| 6 | `gpu_run.sh:48` summary() runs under `set -euo pipefail` inside the trap | minor | Confirmed. `summary()` starts with `set +e; set +o pipefail`; the trap is a named `on_exit` that also releases the clock lock. |
| 7 / 17 / 37 | `bench_main.cu:513` OOM-by-design on `free_bytes/4` vs bench.py `total_memory` (three findings, one defect) | minor | Confirmed. `ws_bytes > dev.total_bytes / 4` (prints free and total); bench.py uses `intermediate_bytes()` (fp16 for `torch_unfused` in fp16) and a truthful note; `test_oom_by_design_rule_agrees_between_harnesses` checks the CONTRACT sweep shapes with the T4 total. |
| 8 | `gpu_run.sh:94` cmake only pip-installed when missing; presets v6 need >= 3.25 | minor | Confirmed. `cmake_ok()` awk version check; pip `cmake>=3.25` when too old; the explicit-configure branch is also taken when cmake is too old. Same check in the notebook PROBE 1 cell. |
| 9 | `bench.py:72` nvcc as `12.8` vs fa_bench `12.8.93` | minor | Confirmed. bench.py parses `V(\d+\.\d+\.\d+)`; both harnesses stamp the full version (finding 18's MAJOR.MINOR suggestion superseded by this one, documented in protocol rule 8). |
| 10 / 31 | `gpu_run.sh:167` clocks never locked although `-lgc 585,585` works on Colab | major | Confirmed. New `lock_clocks` step before the bench: base clock read from `clocks.default_applications.graphics` (fallback: `nvidia-smi -q -d CLOCK` "Default Applications Clocks"), `-lgc` applied, `clock_lock=<mhz>`/`refused` in env.txt, `-rgc` in the EXIT trap, `--clock-locked <mhz>` forwarded to bench.py/gpt_block.py only when locked. bench.py verifies the lock from its in-loop samples (`clock_lock=<mhz>` note) and drops it per row otherwise (`clock=unlocked;clock_lock_failed=<mhz>`). Protocol rule 4, ENV.md and README updated. GPU-unverified: whether `clocks.default_applications.graphics` reports 585 on the Colab T4 (the probe only proved `-lgc 585,585` is accepted). |
| 11 | `bench.py:266` clock sampled after the loop via nvidia-smi; `gpt_block.py:121` samples an idle GPU | major | Confirmed. `ClockSampler` (pynvml `nvmlDeviceGetClockInfo` every 20 ms from a background thread, nvidia-smi fallback) runs from before the timed loop until after the synchronize; `clock_mhz` = median, `clock=<min>-<max>`, `noisy_clock` above 10% (`fa_common.CLOCK_SPREAD`). gpt_block uses the same sampler over its loop. |
| 12 | `fa_common.py:285` unit scaling only knew `{K,M,G,T}{byte,byte/second,Hz}`; `cycle/nsecond`, `usecond`, `byte/cycle` went through unscaled | major | Confirmed. `normalize_unit()` with a complete table (SI prefix on the numerator, `m/u/nsecond` denominators -> `/second`, bare time units -> `second`); anything else with a prefix is withheld (`values[m] = None`, `unparsed_units`), `ncu_to_md` prints `n/a (unit X)`, `roofline.py` refuses the kernel with a printed reason. `profile.sh` records `ncu_print_units=base\|auto` in env.txt and the stage `.meta`. CPU tests with a synthetic auto-unit raw.csv. |
| 13 | `roofline.py:158` / `ENV.md:49` clock of the counters never stated; ENV.md said "at the run's actual clock" | minor | Confirmed. Figure title prints `SM clock during profile: X MHz (ncu --clock-control base)`; ENV.md and PROFILING_PROTOCOL §2 carry the "compare % of roof, not TFLOP/s" sentence. |
| 14 | `bench.py:293` `--sweep` ignores `sweep.causal` | minor | Confirmed. `--causal` defaults to None; `resolve_causals()` takes `sweep.causal` under `--sweep`; gpu_run.sh forwards `--sweep --causal both`; README/runner text says the sweep needs `--sweep`. |
| 15 | `configs.yaml:23` `gbps` labelled "bytes actually moved" but is a minimum model | minor | Confirmed. Renamed "modelled minimum bytes" in CONTRACT §3, protocol rule 6, configs.yaml, bench.py/bench_main.cu comments, with the NCU table named as the measured number. No per-impl pass counts were added (they would be untested claims). |
| 16 | `bench_main.cu:583` 500 ms floor sized from the first launch | minor | Confirmed. `time_loop` warms up, then times three launches and sizes `iters` (unless `--iters` given), like bench.py; rows carry `iters=<n>;warmup=<n>`. |
| 18 | `bench_main.cu:347` provenance: commit `unknown`, `ncu` empty, `clock_mhz` empty, nvcc format | minor | Confirmed. `commit` = `--commit` / `$FA_COMMIT` / `git rev-parse --short HEAD` (+`-dirty`); `ncu` from `ncu --version`; `clock_mhz` from NVML samples at 4 points of the loop (`nvmlDeviceGetClockInfo` via the existing dlopen); `torch` blank by design and the uniform-vs-randn input difference stated in protocol rule 8. gpu_run.sh and profile.sh export `FA_COMMIT`. |
| 19 | `bench_main.cu:592` self-check only head (0,0) rows [0,64) | minor | Confirmed. `self_check` now covers rows `[0, rows/2)` and `[N-rows/2, N)` of heads (0,0) and (B-1,H-1) (GQA-aware kv head), one head-slice at a time so nothing large is copied; note `check=rows<total>`. Compiles; GPU-unverified. |
| 20 | `gpt_block.py:115` violates rules 1-2 silently; stage column typed | minor | Confirmed. One event pair per attention call kept in `Attention.windows`, single synchronize after the loop; `dispatched_stage()` walks `impl_supports` 5..1; notes `warm;no_flush`; the exception is scoped in protocol rule 1. `--stage` argument removed (gpu_run.sh updated). |
| 21 | `profile.sh:99` no commit provenance for profiles | minor | Confirmed. `profiles/<gpu>/stage<N>.meta` (commit, date, ncu version, clock control, dtype, cfgs, print-units, exact ncu command line per report); `ncu_to_md` adds a commit column from it; `test_profile_commit_matches_bench_rows` fails when it differs from the stage's kernel rows. |
| 22 | `test_bench_sanity.py:77` cannot catch formula drift / short-cuts | minor | Confirmed. New assertions: `tflops` recomputed from `4BHN^2D/ms` within 1%; `iters>=100` for every kernel row; `check=`/`max_err=` on our rows; `-dirty` commits rejected; a `pytest` warning when no ROOF parses. **Not done:** the T4 datasheet peaks are not pasted into ENV.md (lead's call; the numbers in the finding are unsourced here). |
| 23 | `bench.py:198` refused backend not a row | minor | Confirmed. A `backend_refused:<msg>` row (empty `ms_*`) per skipped baseline; sanity test accepts it; `bench_to_md` renders `refused` and "no baseline row" when the denominator is absent; `--baselines none` prints a warning. |
| 24 | `fa_common.py:114` `latest_rows` file order shadows newer baselines | minor | Confirmed. Rows carry `run=<utc>`; `latest_rows` keeps the greatest stamp (file order for ties/unstamped); `bench_to_md` appends "(baseline @ `<sha>`)" when the SDPA row's commit differs from the kernel row's. CPU test. |
| 25 | `bench.py:75` commit without a dirty check | minor | Confirmed. `fa_common.git_commit()` (`-dirty` when a tracked file outside `bench/results`/`profiles` differs; untracked files ignored because the run creates them) used by bench.py/gpt_block.py; same rule in gpu_run.sh `$SHA` and in `bench_main.cu::git_commit`; sanity test rejects `-dirty`. |
| 26 | `RESEARCH.md:260` "`--clock-control` defaults to boost"; PROFILING §2 "~110 launches" | minor | Confirmed (ncu CLI docs: base is the default). Corrected; §2 now says 23 matching launches with `--iters 12`, `--launch-skip 10` profiles the 11th (last warmup). |
| 27 | `bench_to_md.py:111` StopIteration on a smoke-only stage0.csv | major | Confirmed (reproduced in the new test). Rows outside `IMPL_ORDER` are dropped before rendering; a shape with no rendered impl is skipped; a results tree with only such rows renders the empty text. `test_bench_to_md_smoke_only_stage0_renders_empty_block`. |
| 28 | `bench.py:418` / `gpt_block.py` never append `noisy` | major | Confirmed. Both append `noisy` at `p90/p10 >= fa_common.NOISE_RATIO` (1.15); `test_noise_constants_match_bench_main` pins `kNoiseRatio`/`kClockSpread` to the Python constants. |
| 29 | `INTERVIEW.md:29` / `NUMERICS.md:52` describe FA's guards, not `softmax_math.cuh`'s | major | Confirmed (`rescale_alpha` -> 0, `finalize_inv_l` -> 0, no `Is_first`). Both passages rewritten to the repo's guards, FA kept as the citation with the equivalence stated in one sentence. |
| 30 | Stale TBD/UNVERIFIED cells contradicted by `.research/colab_probe_2026-09-02.txt` | major | Confirmed. Probe pasted into ENV.md (Colab column), STAGES.md PROBE L/1/2 rows, RESEARCH.md §5/§8 and PROFILING §3/§5 (nsys absent on Colab); `sm__throughput...` and `gpu__time_duration.sum` enabled in `ncu_metrics.txt`; `gpu__compute_memory_throughput...` stays UNVERIFIED because the probe's grep output does not list it; probe committed as `bench/results/t4/probe_2026-09-02.txt`. |
| 32 | `INTERVIEW.md:40` Theorem-2 factor at d=128 | major | Confirmed. Stated numerically (`M = 16K` fp32 elements on T4: 4x at d=64, 1x at d=128; H100 for contrast), the practical collapse attributed to L2 with `lts__t_bytes.sum` recorded next to `dram__bytes.sum`; no uncited L2 size introduced. |
| 33 | `CONTRACT.md:29` impl vocabulary vs `smoke`/`gpt_*` | minor | Confirmed. CONTRACT §3 and the protocol's schema section list them; `fa_common.EXTRA_IMPLS` replaces the ad-hoc test exception. |
| 34 | `NUMERICS.md:70` cites `CAUSAL_ALIGN` and three test names that do not exist | minor | Confirmed. `CAUSAL_ALIGN = "top_left"` added to `reference.py` (referenced from `_keep_mask`'s docstring); the three names replaced with the real ones. |
| 35 | `tile_config.h:19` padding rationale example matches no tile; occupancy not derived | minor | Confirmed. Comment rewritten (128 B-multiple strides, +16 B = 4 banks, conflict-free for 16-B chunk walks, 4-way for scalar walks, why not +1); `kBlocksPerSmBySmem*` / `kWarpsPerSmBySmem*` added with `static_assert`s for the T4 numbers; INTERVIEW §6 quotes the constants. Header host-compiles. |
| 36 | `README.md:102` "runs the benchmark sweep"; line 13 "checked with opcheck" | minor | Confirmed. Both sentences corrected. |
| 37 | `RESEARCH.md:339` alexarmbr date 2024-08-10 "not in the bundle" | minor | **Skipped as wrong**: the URL in `.research/tech-reality-check-4-stage-cuda-flashattentio.json` is `alexarmbr.github.io/2024/08/10/...`, so the date is from the bundle; the table cell now says "(from the URL path)". |
| 39 | `bench.py:360` deprecated `allow_tf32` booleans | minor | Confirmed. `pin_fp32_ieee()` sets `fp32_precision = "ieee"` when the attribute exists (torch 2.11 on Colab), else the booleans; the choice is written to env.txt. |
| 40 | `gpu_run.sh:135` sanitizer evidence file appends forever | minor | Confirmed. Truncated at the start of the step; each tool's block is prefixed with commit, date, stage, dtype and rc. |

Files changed by this pass: `scripts/gpu_run.sh`, `scripts/fa_common.py`, `scripts/bench_to_md.py`,
`scripts/ncu_to_md.py`, `scripts/roofline.py`, `scripts/profile.sh`, `scripts/local_compile.sh`,
`scripts/ncu_metrics.txt`, `.github/scripts/ptxas_summary.py`, `bench/bench.py`, `bench/gpt_block.py`,
`bench/bench_main.cu`, `bench/configs.yaml`, `include/fa/tile_config.h`, `python/fused_attn/ops.py`,
`python/fused_attn/reference.py`, `tests/test_bench_sanity.py`, `tests/test_fa_common_cpu.py` (new),
`colab/runner.ipynb`, `CONTRACT.md`, `README.md`, `docs/{INTERVIEW,NUMERICS,STAGES,ENV,PROFILING_PROTOCOL,BENCHMARK_PROTOCOL,RESEARCH}.md`,
`bench/results/t4/probe_2026-09-02.txt` (new, copy of the probe). Not committed.
