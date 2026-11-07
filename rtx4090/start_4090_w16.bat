@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set TEMP=D:\Temp
set TMP=D:\Temp
set STRATA_ARENA_PIN_GIB=6
set STRATA_ARENA_RAM_FALLBACK=0
set STRATA_ARENA_RAM_WAIT_S=0
set STRATA_ADAPT_OVERLAP=1
set STRATA_ADAPT_OVERLAP_RESIDENT=1
set STRATA_IQ3S_MT1=1
set STRATA_IQ_MT_MIN=1
set STRATA_DISABLE_EMBEDDINGS=1
set STRATA_ADMIT_SHORT_FIRST=8192
set STRATA_ADMIT_MAX_WAIT_S=30
set OPENBLAS_NUM_THREADS=4
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4

nvidia-smi -pl 410 >nul 2>&1
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 5 >nul 2>&1
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 90 >nul 2>&1
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE 4 >nul 2>&1
powercfg /setactive SCHEME_CURRENT >nul 2>&1

cd /d D:\Strata-v0.2.12
E:\Strata\.venv\Scripts\python.exe D:\Strata-v0.2.12\repo\serve\server.py --engine strata --config D:\Strata-v0.2.12\strata-iq3_s-4090-w16.json --port 8080 --host 0.0.0.0 >> D:\Strata-v0.2.12\strata_server.log 2>&1
