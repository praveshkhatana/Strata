# Strata 0.1.59.1 — RTX 4090 (24 GB) + i9-14900K (64 GB RAM) Ubuntu Linux Tuning & Findings

Production setup, code modifications, kernel/hardware guardrails, and reproducible benchmarks on `Qwen3.8-Flash-Next-IQ3_S` (`200k` context, `DFlash2`, `parallel=3`).

---

## 1. Hardware & OS Guardrails

- **GPU**: NVIDIA GeForce RTX 4090 (`sm_89`, `128` SMs, `24564 MiB` VRAM)
  - Power limit capped at **`410 W`** via `rtx4090-linux/systemd/nvidia-power-limit.service`.
- **CPU**: Intel Core i9-14900K (24 cores / 32 threads: P-cores CPUs `0..15`, E-cores CPUs `16..31`)
  - P-cores capped to **`4.80 GHz`** (`4800000`) and E-cores to **`4.40 GHz`** (`4400000`) with `energy_performance_preference=performance` via `rtx4090-linux/systemd/intel-cpu-safe-cap.service`.
- **RAM & Virtual Memory**: 64 GB (`62 GiB` usable) DDR5 + `32 GiB` NVMe swapfile (`/mnt/nvme/swapfile`, `sw,pri=10` in `/etc/fstab`) + `rtx4090-linux/systemd/99-ai-server-vm.conf` (`vm.swappiness=100`, `vm.vfs_cache_pressure=50`, `vm.page-cluster=0`).
- **IOMMU Passthrough (`iommu=pt`)**: Enabled in both `/etc/default/grub` (`GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt"`) and `/etc/grub.d/40_custom` (`gnulinux-gpu-unlock-4090`, preserving `gpu_unlock.targets=0000:01:00.0`). Verified active post-reboot (`/sys/kernel/iommu_groups/*/type` switched from `DMA-FQ` to `identity`).

---

## 2. Strata 4-Tier Hybrid Architecture & Root Causes Found

Strata splits `Qwen3.8-Flash-Next-IQ3_S` (~46.8 GiB routed experts + dense/attention/GDN/shared layers) across 4 tiers:
1. **Always-on GPU (VRAM)**: Dense/attention/GDN/router/shared experts + `32768 / 200000` KV resident window (`1.29 GiB` pinned RAM backup) + `DFlash2` (`492 MiB` + `81.2 MiB` head).
2. **VRAM Hot Expert Cache (`R4` Token Graph Hit Path)**: Most frequently activated routed experts stay resident in RTX 4090 VRAM (`7118` slots = `13.50 GiB` with `--vram-reserve-mib 1300 --vram-elastic`), and prompt prefill borrows `3466` cache slots (`6.52 GiB`) for `16384` chunks.
3. **Host RAM Expert Pool + CPU/PCIe Tail**: Remaining experts (`39.86 GiB`) stay 100% page-locked (`mlock` + `cudaHostRegister`) in host DDR5 RAM (`--resident-experts`, `0` file reads, `0.0 swaps/window`) and are processed by 15 AVX-2 pool workers + adaptive PCIe share.
4. **DFlash2 Speculative Decoding (`q4_0-selq4.pack`)**: CUDA graphs capture windows up to `8` tokens with `--spec 6 --draft-cap 6 --draft-conf 0.50` and `STRATA_ROUTE_TAIL_SKIP=0` (exact greedy routing preserved).

### Why the Initial Linux Config Bottlenecked (~70–91 tok/s)
1. **`--expert-cache 4750` artificially capped VRAM experts**: It restricted cache to `6203` slots (`11.78 GiB`, leaving `3027 MiB` free VRAM unused) and shrunk prompt borrowed slots to `1871` (`3.51 GiB`, forcing `8192` prefill chunks). Removing `--expert-cache 4750` lets `--vram-reserve-mib 1300 --vram-elastic` auto-allocate **`7118` slots (`13.50 GiB`)** and **`3466` borrowed prefill slots (`6.52 GiB`)**.
2. **`STRATA_ARENA_PIN_GIB=6` forced unpinned buffered experts**: Only `5 GiB` (`6` slices) were pinned; the rest fell back through file cache. Removing `STRATA_ARENA_PIN_GIB` and enabling **`--resident-experts`** pinned **`39.86 GiB`** cleanly in RAM.
3. **`--pool-affinity p-cores --pool-workers 7` under-utilized P+E cores and over-queued PCIe**: Switching to **`--pool-affinity all --pool-workers 15 --aux-cpus auto --pcie-frac 0.10`** places `15` workers on CPUs `2,4,6,8,10,12,14,16..23`, host thread on CPU `0`, and auxiliary/control threads on CPUs `1,24..31`, cutting `GPU-reach wait` while avoiding PCIe congestion.
4. **`STRATA_DF2_GEMM4=1` and un-tuned Ada MMVQ/GR paths**: Disabled RTX 4090 Q4 auto-dispatch and left GR reads/Upmix/QSA selection un-fused.

---

## 3. Repository Code Changes (on top of Strata `0.1.59.1`)

Branch `strata-rtx4090-opt` / `4090-linux` carries two custom commits on upstream `main`:

1. **`57e35d93`** — *Port custom RTX 4090 fixes, PR #5203 QSA F16, and batch PCIe/staging hooks onto upstream main*:
   - `cmake/strata_gpu_engine.cmake`, `src/kernels/cuda/qsa_select.cu` (`STRATA_SELECT_F16=1`, ~5.0x faster `qsa_select_bench` at 131K: `1.578 -> 0.318 ms`)
   - `include/strata/core/verify.hpp`, `src/core/verify.cpp`, `src/program/generate.cpp` (`kStagingBlobs = 32`, batch PCIe/staging hooks)
   - `include/strata/kernels/cpu/pool.hpp`, `src/kernels/cpu/pool.cpp` (`QuantTask` parallel intermediate quant)
   - `serve/frontend.py`, `serve/server.py`, `src/kernels/cuda/iq_kernels.cu` (`mcp__` tool alias normalization, `sampling.force` support)
2. **`2549a342`** — *Optimize RTX 4090 MMVQ/DFlash2 tables and 24GB DFlash2 pack selection*:
   - `src/kernels/cuda/native_mmvq.cu`: Added RTX 4090 (`8901`) interleaved decode tables (`kIlRows58_8901`, `kIl1Rows_8901`, `kIlRows_8901`, `kIlPf_8901`, enabled by default unless `STRATA_MMVQ_IL_8901=0`). Measured isolated A/B gain (`roundD1b` vs `roundE6` with `8901=0`): `+8.62` solo_short, `+10.14` solo_long_cold, `+26.15` solo_long_warm, `+21.07` batch_3 (`153.26` vs `132.19`).
   - `src/kernels/cuda/dflash2.cu`: Added `ada_big` (`maj == 8 && mnr == 9 && sms >= 100`) to `df2_q4_auto(int rows)` (`var 20` if `rows <= 256 && rows % 32 == 0`, `var 31` if `rows % 8 == 0`, else `17`; requires `STRATA_DF2_GEMM4=0`).
   - `strata_control/drafting.py`: Preserves explicit `--draft-model` / `tuned_ok()` pack and `engine_reads_selector_q4()` on 24 GB cards when stock `pack_ready()` returns false.

---

## 4. Benchmark Progression (`temperature=0.0, top_k=1`, Exact Greedy Parity)

| Scenario | Old (`bench_old`) | New Base (`0.1.59.1` + Old JSON) | Split Tune (`roundC4`) | Peak Decode (`roundD1b`) | Live Deployed (`roundE7`: `D1b` + `HC_UPMIX=1`) |
|---|---:|---:|---:|---:|---:|
| `solo_short` (47->180) | `91.25` | `71.80` (`0.44s` TTFT) | `124.83` (`0.38s`) | **`130.32`** (`0.37s`) | `121.83` (`0.42s`) |
| `solo_long_cold` (39.6K) | `100.78` (`21.37s` TTFT) | `70.46` (`20.72s` TTFT) | `114.55` (`10.84s` TTFT) | **`126.67`** (`10.84s` TTFT) | `124.64` (**`10.58s` TTFT, 2.02x faster**) |
| `solo_long_warm` (39.6K) | `120.62` | `80.12` (`4.42s` TTFT) | `137.97` (`2.94s` TTFT) | **`151.82`** (`2.94s` TTFT) | `140.28` (`2.96s` TTFT) |
| `batch_2` (2 concurrent) | `119.31` (`3.59s` wall) | `95.36` (`4.04s` wall) | `126.44` (`3.12s` wall) | `134.83` (`2.97s` wall) | **`137.06`** agg (`71.38`/slot, `3.08s` wall) |
| `batch_3` (3 concurrent) | `118.34` (`4.90s` wall) | `125.32` (`4.43s` wall) | `130.12` (`4.23s` wall) | **`153.26`** (`3.75s` wall) | `147.26` agg (`51.97`/slot, `3.90s` wall) |

> **Tip**: `roundE7` gives the fastest cold 39.6K prefill (`10.58s`) and `batch_2` (`137.06`). If solo decode throughput matters more than `-260 ms` cold prefill, set `"STRATA_HC_UPMIX": "0"` in `strata-iq3_s-linux.json` (`roundD1b` profile).

### Rejected Experiments (Do Not Re-Apply)
- `STRATA_PF_FUSED=1` variants (`D2a/D2b/D2c/E2c/E5`): Cut cold TTFT to `10.25-10.30s` but reserved `~1.44 GiB` VRAM, shrinking borrowed prompt slots `3466 -> 2703` and reducing decode throughput.
- `STRATA_EXPDEC=0` (`D1c`): Slower across all 5 cases even though microbench looked favorable.
- `STRATA_FUSE_WAIT=1 + STRATA_FUSE_HEAD_GR=1` (`D1d`): Slower than `STRATA_FUSE_WAIT=2`.
- `STRATA_FS_SLOTS=2` (`E1`): Slower (`125.30 / 108.45 / 130.12 / 123.71 / 138.00`).
- `--pcie-frac 0.00 / 0.15 / 0.30`: All worse overall than `--pcie-frac 0.10`.
- `--kv-resident 49152` (`E4`): Slower than `32768` (`32768` already yields `95.57%-99.32%` VRAM KV reads).
- `--spec 8 --draft-cap 8 --draft-conf 0.35, MAXX=6, DEFER=1` (`E2a`) and `DF2_MAXX=2, CONF=0.55, DEFER=1` (`E2b`): Lower decode throughput.

---

## 5. Drafter Note: `DFlash2` vs `DSpine` vs `MTP`

- **DSpine** (Adjacent Causal Injection block-diffusion drafter) is **not** currently implemented in Strata `0.1.59.1`.
- Strata supports **DFlash2** (`qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack`), native **MTP**, and suffix/prompt-lookup. On RTX 4090 with `df2_q4_auto` + `STRATA_DF2_GEMM4=0`, `DFlash2` adds only `~1.00 ms/window` draft overhead.

---

## 6. Clean Setup & Verification From Scratch

```bash
# 1. Clone branch
git clone -b 4090-linux https://github.com/praveshkhatana/Strata.git /mnt/ssd/strata/repo
cd /mnt/ssd/strata/repo

# 2. (Optional, Ubuntu host caps & VM) Install systemd units and sysctl once as root
sudo cp rtx4090-linux/systemd/nvidia-power-limit.service /etc/systemd/system/
sudo cp rtx4090-linux/systemd/intel-cpu-safe-cap.service /etc/systemd/system/
sudo cp rtx4090-linux/systemd/99-ai-server-vm.conf /etc/sysctl.d/
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-power-limit.service intel-cpu-safe-cap.service
sudo sysctl --system

# 3. Build Strata (sm_89, 16-row windows, tests) and Vision
./rtx4090-linux/build_4090_linux.sh

# 4. Copy runtime launcher & tuned config into /mnt/ssd/strata
cp rtx4090-linux/start_strata_linux.sh /mnt/ssd/strata/
cp rtx4090-linux/strata-iq3_s-linux.json /mnt/ssd/strata/

# 5. Verify CUDA parity, ctest, and Python drafting tests (stop server before GPU tests)
ctest --test-dir /mnt/ssd/strata/build --output-on-failure \
  -R "mmvq|verify_batch|qsa_select|dflash2|native_expert|nv_defaults|pcie_share|draft_|arch_defaults"
STRATA_GR_V3=1 STRATA_GR_V3_PF=133 /mnt/ssd/strata/build/gr_parity
/mnt/ssd/strata/build/gr_upmix_parity
STRATA_B6_MMVQ_ROWS=1 /mnt/ssd/strata/build/b6_mmvq_bench
/mnt/ssd/strata/build/mmvq_il_parity
/mnt/ssd/strata/build/qsa_merge_daily_parity
/mnt/ssd/strata/build/native_expert_parity --synthetic IQ3_S/IQ4_NL

/mnt/ssd/strata/.venv/bin/python -m unittest \
  tests/terminal/test_drafting.py \
  tests/terminal/test_drafting_tuned.py \
  tests/terminal/test_drafting_single.py \
  tests/terminal/test_drafting_streaming_class.py \
  tests/terminal/test_drafting_card_window.py \
  tests/terminal/test_drafting_auto_mtp.py \
  tests/terminal/test_drafting_selector_q4.py \
  serve/test_drafting_retry.py \
  serve/test_draft_layer.py

# 6. Launch server (:8080)
setsid nohup /mnt/ssd/strata/start_strata_linux.sh </dev/null >/dev/null 2>&1 &
curl -s http://127.0.0.1:8080/health
```
