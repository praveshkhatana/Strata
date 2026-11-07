#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/cuda/bin:$PATH"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build-spark-w16}"
STAGE_DIR="${STAGE_DIR:-$HOME/strata-engine-w16}"

cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_WINDOW_MAX_T=16 \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_PORTABLE=ON \
  -DSTRATA_Q6K_EXPERTS=ON \
  -DSTRATA_MMQ_KQUANTS=ON \
  -DCMAKE_INSTALL_RPATH='$ORIGIN' \
  -DCMAKE_CUDA_ARCHITECTURES='121-real;121-virtual'

nice -n 19 cmake --build "$BUILD_DIR" --target window_limits_test strata strata_kq -j 4
"$BUILD_DIR/window_limits_test"

mkdir -p "$STAGE_DIR"
cp "$BUILD_DIR/strata" "$STAGE_DIR/strata.new"
cp "$BUILD_DIR/libstrata_kq.so" "$STAGE_DIR/libstrata_kq.so"
mv "$STAGE_DIR/strata.new" "$STAGE_DIR/strata"
echo "Staged $STAGE_DIR/strata (sm_121 16-row build)"
