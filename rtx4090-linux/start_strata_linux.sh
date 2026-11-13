#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export TMP=/mnt/ssd/tmp
export TEMP=/mnt/ssd/tmp
export TMPDIR=/mnt/ssd/tmp
export HF_HOME=/mnt/ssd/.cache/huggingface
export XDG_CACHE_HOME=/mnt/ssd/.cache
export PIP_CACHE_DIR=/mnt/ssd/.cache/pip
export PATH=/mnt/ssd/cuda-13.0/bin:/mnt/ssd/strata/.venv/bin:$PATH
export LD_LIBRARY_PATH=/mnt/ssd/strata/build:/mnt/ssd/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /mnt/ssd/strata
exec /mnt/ssd/strata/.venv/bin/python /mnt/ssd/strata/repo/serve/server.py \
  --engine strata \
  --config /mnt/ssd/strata/strata-iq3_s-linux.json \
  --port 8080 \
  --host 0.0.0.0 >> /mnt/ssd/strata/strata_server.log 2>&1
