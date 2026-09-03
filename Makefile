# Developer entry points. Everything GPU-related runs remotely (scripts/gpu_run.sh); these
# targets are what runs on the Mac and in CI.
PYTHON ?= python3

.PHONY: help cpu-test lint docs local-compile reproduce

help:
	@echo "make cpu-test       configure+build+ctest the cpu-only preset (fp64 reference, no CUDA)"
	@echo "make lint           ruff check/format + clang-format --dry-run on C++/CUDA"
	@echo "make docs           regenerate README tables and the roofline PNG from committed CSVs"
	@echo "make local-compile  nvcc compile loop in Docker, no GPU needed:"
	@echo "                    build/docker/build.log + build/docker/ptxas.txt"
	@echo "make reproduce      print the Colab / VM sequence that produces every number in the repo"

# cmake >= 3.25 and ninja: `brew install cmake ninja` on the Mac.
cpu-test:
	cmake --preset cpu-only
	cmake --build --preset cpu-only
	ctest --preset cpu-only --output-on-failure

# git ls-files keeps build/ and .research/ out; xargs -r skips clang-format when there
# is nothing to check. .cu/.cuh are formatted as C++ (CONTRACT §8).
lint:
	ruff check .
	ruff format --check .
	git ls-files -z -- '*.cu' '*.cuh' '*.cpp' '*.h' '*.hpp' | xargs -0 -r \
		clang-format --dry-run --Werror --style=file

docs:
	$(PYTHON) scripts/bench_to_md.py
	$(PYTHON) scripts/ncu_to_md.py
	$(PYTHON) scripts/roofline.py

local-compile:
	bash scripts/local_compile.sh

reproduce:
	@echo "Every number in this repo is produced by the following sequence (PLAN.md, Track A/B):"
	@echo ""
	@echo "  Mac (no GPU):"
	@echo "    make cpu-test                       # fp64 reference + tests"
	@echo "    pytest -q -m 'not gpu'              # CPU model of the online softmax, docs staleness"
	@echo "    make local-compile                  # optional: nvcc syntax/ptxas check in Docker"
	@echo ""
	@echo "  Colab (T4, free tier) -- open colab/runner.ipynb, runtime = T4, secret GH_PAT set:"
	@echo "    cell 1: environment probe           -> bench/results/t4/env.txt"
	@echo "    cell 2: ncu go/no-go                -> counters PASS or degraded-mode reason"
	@echo "    cell 3: BRANCH=<stage branch> STAGE=<N> MODE=full ->"
	@echo "            bash scripts/gpu_run.sh --stage N --gpu t4 --mode full --push"
	@echo "            (cmake --preset colab; ctest; compute-sanitizer on fa_bench --smoke;"
	@echo "             pytest -m gpu; bench/bench.py; scripts/profile.sh; git push results)"
	@echo ""
	@echo "  Optional sm_80+ VM (ssh host 'gpu'):"
	@echo "    bash scripts/gate_gpu.sh            # PROBE 4: ncu works as user or sudo, else stop"
	@echo "    git push && ssh gpu 'cd ~/fused-attention-cuda && git pull --ff-only &&"
	@echo "                         bash scripts/gpu_run.sh --stage N --gpu a10 --mode full'"
	@echo "    rsync -az gpu:~/fused-attention-cuda/bench/results/ bench/results/"
	@echo "    rsync -az gpu:~/fused-attention-cuda/profiles/ profiles/"
	@echo ""
	@echo "  Back on the Mac after each run:"
	@echo "    git pull --ff-only && make docs && pytest -q -m 'not gpu'"
	@echo ""
	@echo "  Direct fa_bench usage on any CUDA box:"
	@echo "    cmake --preset colab && cmake --build --preset colab"
	@echo "    ./build/colab/bench/fa_bench --smoke"
	@echo "    ./build/colab/bench/fa_bench --stage 3 --cfg C64 --dtype fp32 \\"
	@echo "        --out bench/results/t4/stage3.csv --commit \$$(git rev-parse --short HEAD)"
