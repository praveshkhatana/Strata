#!/usr/bin/env bash
set -euo pipefail

# Reproducible Linux build script for RTX 4090 (sm_89, 128 SMs) + i9-14900K
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_DIR="${BASE_DIR:-/mnt/ssd/strata}"
CUDA_HOME="${CUDA_HOME:-/mnt/ssd/cuda-13.0}"
BUILD_DIR="${BUILD_DIR:-$BASE_DIR/build}"
VISION_BUILD_DIR="${VISION_BUILD_DIR:-$BASE_DIR/build-vision}"
LLAMA_DIR="${LLAMA_DIR:-$BASE_DIR/llamacpp}"
JOBS="${JOBS:-16}"

export PATH="$CUDA_HOME/bin:$PATH"

cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="89-real" \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_WINDOW_MAX_T=16 \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_Q6K_EXPERTS=ON \
  -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_MMQ_TAB=ON \
  -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_PORTABLE=OFF

cmake --build "$BUILD_DIR" -j"$JOBS"

if [ -d "$LLAMA_DIR" ] && [ "${BUILD_VISION:-1}" = "1" ]; then
  cmake -S "$SRC_DIR/tools/vision" -B "$VISION_BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="89-real" \
    -DLLAMA_DIR="$LLAMA_DIR" \
    -DSTRATA_VISION_CUDA=ON \
    -DSTRATA_PORTABLE=OFF

  cmake --build "$VISION_BUILD_DIR" --target strata-vision -j"$JOBS"
fi

echo "Build complete:"
echo "  Engine : $BUILD_DIR/strata"
if [ -x "$VISION_BUILD_DIR/bin/strata-vision" ]; then
  echo "  Vision : $VISION_BUILD_DIR/bin/strata-vision"
fi
