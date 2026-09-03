#!/usr/bin/env bash
# Local nvcc compile loop on a machine with no NVIDIA GPU (the Mac).
#
# WHY: a syntax or type error, or a register-spill report, should cost seconds here, not a
# Colab round trip. nvcc does not need a GPU to compile, and nvidia/cuda devel images ship
# linux/arm64 variants, so the whole build runs inside a container with the repo mounted.
#
# usage: scripts/local_compile.sh            # archs default to the CI list
#        FA_ARCHS="75-real" scripts/local_compile.sh
# env:   FA_ARCHS            CMAKE_CUDA_ARCHITECTURES value (default: CI list below)
#        FA_DOCKER_PLATFORM  default linux/arm64 (use linux/amd64 on an x86 host)
# out:   build/docker/build.log  full configure+build output
#        build/docker/ptxas.txt  the ptxas -v report: every "ptxas info/warning" line plus the
#                                indented spill/stack line (registers / spills / smem per kernel)
# exit:  non-zero when configure or compile fails (docker's own status is propagated).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="fa-nvcc:12.8.1"
PLATFORM="${FA_DOCKER_PLATFORM:-linux/arm64}"
FA_ARCHS="${FA_ARCHS:-75-real;80-real;86-real;89-real;90-real;90-virtual}"
OUT="$ROOT/build/docker"
LOG="$OUT/build.log"
PTXAS="$OUT/ptxas.txt"

if ! command -v docker >/dev/null 2>&1; then
    echo "local_compile: docker not found. Install colima+docker (brew install colima docker;" >&2
    echo "colima start --cpu 4 --memory 6) or use the CI compile job instead." >&2
    exit 2
fi
if ! docker info >/dev/null 2>&1; then
    echo "local_compile: docker daemon not running (try: colima start)" >&2
    exit 2
fi

# Build the image once; rebuild only when the Dockerfile changes (docker layer cache handles
# that) or the image is missing.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "local_compile: building $IMAGE for $PLATFORM (one-time, downloads the CUDA devel image)"
    docker build --platform "$PLATFORM" -f "$ROOT/docker/Dockerfile.nvcc" -t "$IMAGE" "$ROOT"
fi

mkdir -p "$OUT"
echo "local_compile: archs=$FA_ARCHS platform=$PLATFORM -> $LOG"

# The `docker` preset inherits the CI arch list and pins binaryDir to build/docker; the
# explicit -D on the command line lets FA_ARCHS narrow it (a single arch compiles ~6x faster).
set +e
docker run --rm --platform "$PLATFORM" \
    -v "$ROOT:/w" -w /w -e FA_ARCHS="$FA_ARCHS" "$IMAGE" bash -c '
        set -euo pipefail
        nvcc --version | tail -1
        cmake --fresh --preset docker -DCMAKE_CUDA_ARCHITECTURES="$FA_ARCHS"  # --fresh: a cached arch list must never win over FA_ARCHS
        cmake --build --preset docker
    ' 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

# The spill line is indented and has no `ptxas` prefix (same pattern as build.yml / gpu_run.sh).
grep -E '^ptxas (info|warning)|bytes spill|bytes stack frame' "$LOG" > "$PTXAS" || true
echo "local_compile: $(grep -c . "$PTXAS" 2>/dev/null || echo 0) ptxas lines -> $PTXAS"
if grep -qE "spill stores|spill loads" "$PTXAS"; then
    echo "local_compile: spill report (0 bytes = good):"
    grep -E "spill" "$PTXAS" | sort | uniq -c | sort -rn | head -20
fi

if [[ "$status" -ne 0 ]]; then
    echo "local_compile: FAILED (status $status); first errors:" >&2
    grep -nE "error|Error" "$LOG" | head -20 >&2 || true
fi
exit "$status"
