# fused-attention-cuda build.
#
# WHY this shape:
#   * project() starts with CXX only so `cmake --preset cpu-only` configures on a Mac
#     with no CUDA toolkit: the fp64 CPU reference and its ctest build anywhere.
#   * CUDA is enabled behind FA_ENABLE_CUDA (default ON) and only AFTER the default
#     architecture is chosen, because CMAKE_CUDA_ARCHITECTURES is consumed when the
#     language is enabled, not when a target is created.
#   * Every .cu under src/ is listed in sources.txt (repo root); python/setup.py reads
#     the same file, so the CMake and torch-extension source lists cannot drift.
#   * Release never carries -G: device debug codegen changes register allocation and
#     the committed ncu profiles would describe a kernel nobody benchmarks.
cmake_minimum_required(VERSION 3.24) # `native` CUDA arch value needs 3.24

project(fused_attn LANGUAGES CXX)

option(FA_ENABLE_CUDA "Build the CUDA kernels and fa_bench (OFF = CPU reference + tests only)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- fp64 CPU reference: always built, no CUDA. The ctest oracle for small shapes ----
add_library(fa_cpu_reference STATIC src/reference/cpu_reference.cpp)
target_include_directories(fa_cpu_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/reference)
target_compile_options(fa_cpu_reference PRIVATE -Wall -Wextra)

enable_testing()
add_executable(test_cpu_reference src/reference/test_cpu_reference.cpp)
target_link_libraries(test_cpu_reference PRIVATE fa_cpu_reference)
target_compile_options(test_cpu_reference PRIVATE -Wall -Wextra)
add_test(NAME test_cpu_reference COMMAND test_cpu_reference)

if(NOT FA_ENABLE_CUDA)
  message(STATUS "fused_attn: FA_ENABLE_CUDA=OFF -> CPU reference + test_cpu_reference only")
  return()
endif()

# ---- CUDA ----
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
  # T4 (sm_75) is the primary GPU. "-real" = SASS only, no embedded PTX: smaller binary,
  # no JIT at first launch, and the profiled code is exactly what ptxas reported.
  # Presets override this (vm: native; ci: the five-arch list + sm_90 PTX).
  set(CMAKE_CUDA_ARCHITECTURES "75-real" CACHE STRING "CUDA architectures" FORCE)
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Guard: -G (device debug) is forbidden in Release. It disables device optimisation, so
# every register/spill/stall number from such a build would be meaningless.
string(TOUPPER "${CMAKE_BUILD_TYPE}" _fa_build_type)
if(_fa_build_type STREQUAL "RELEASE")
  foreach(_flags IN ITEMS "${CMAKE_CUDA_FLAGS}" "${CMAKE_CUDA_FLAGS_RELEASE}")
    if(_flags MATCHES "(^| )-G( |$)")
      message(FATAL_ERROR "fused_attn: -G is forbidden in Release builds (flags: '${_flags}')")
    endif()
  endforeach()
endif()

# Device flags for every CUDA target (Release already carries -O3 -DNDEBUG from CMake):
#   -lineinfo                 source-line correlation in ncu/nsys, no codegen impact
#   -Xptxas=-v                registers / spills / smem per kernel in the build log
#                             (gpu_run.sh greps "ptxas info" into bench/results/<gpu>/)
#   --expt-relaxed-constexpr  constexpr helpers in include/fa/tile_config.h from device code
set(FA_CUDA_DEVICE_FLAGS -lineinfo -Xptxas=-v --expt-relaxed-constexpr)

# FA_BUILD_INFO: stamped into fa_bench so every CSV row can name what it was built with.
# The nvcc version itself comes from __CUDACC_VER_MAJOR__/MINOR__/BUILD__ inside the .cu.
string(REPLACE ";" "," _fa_archs_str "${CMAKE_CUDA_ARCHITECTURES}")
set(
  FA_BUILD_INFO_DEFS
  FA_BUILD_INFO=1
  "FA_CUDA_ARCHS=\"${_fa_archs_str}\""
  "FA_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\""
  "FA_CMAKE_VERSION=\"${CMAKE_VERSION}\""
)

# ---- fa_kernels: every .cu listed in sources.txt (one path per line, relative to root) ----
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/sources.txt _fa_sources_rel REGEX "^[^#].*\\.cu[ \t]*$")
set(_fa_sources "")
foreach(_rel IN LISTS _fa_sources_rel)
  string(STRIP "${_rel}" _rel)
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_rel})
    message(FATAL_ERROR "sources.txt lists '${_rel}' but that file does not exist")
  endif()
  list(APPEND _fa_sources ${CMAKE_CURRENT_SOURCE_DIR}/${_rel})
endforeach()
if(NOT _fa_sources)
  message(FATAL_ERROR "sources.txt is empty; it must list every .cu under src/")
endif()
message(STATUS "fused_attn: CUDA archs=${CMAKE_CUDA_ARCHITECTURES}; sources=${_fa_sources_rel}")

add_library(fa_kernels STATIC ${_fa_sources})
target_include_directories(fa_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(fa_kernels PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:${FA_CUDA_DEVICE_FLAGS}>")
target_compile_definitions(fa_kernels PUBLIC ${FA_BUILD_INFO_DEFS})
target_link_libraries(fa_kernels PUBLIC CUDA::cudart)

# ---- fa_bench: torch-free harness (bench + smoke). Lands in <build>/bench/fa_bench so the
# plain `cmake -S . -B build` used by scripts/gpu_run.sh finds ./build/bench/fa_bench.
add_executable(fa_bench bench/bench_main.cu)
target_link_libraries(fa_bench PRIVATE fa_kernels fa_cpu_reference ${CMAKE_DL_LIBS})
set_target_properties(fa_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bench)

# Round-trip proof: build -> run -> CSV row. Needs a GPU, so it only exists in CUDA builds.
add_test(NAME test_smoke COMMAND fa_bench --smoke)
