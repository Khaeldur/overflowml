# Field report: OverflowML on a shared RTX 5090 (MedSimAI, 2026-09-30)

Written for the OverflowML roadmap (all users, not only MedSimAI). Everything below was **measured** on the owner's PC
unless marked *estimate*. Source data: MedSimAI repo `docs/reports/2026-09-30_gpu_throughput_tryouts.md` (raw JSON
`data_rnd/bench/gpu_throughput_20260930/`), `engine/gpu/overflow.py` (the gate MedSim runs today).

Host: RTX 5090 32 GB (31.8 GB usable in WSL2), Ryzen 9 9950X3D (15 cores / 30 threads visible), WSL2 Ubuntu 24.04,
157 GB RAM visible now (`.wslconfig memory=128GB` → 128 GB after the next WSL restart), `networkingMode=mirrored`,
`ulimit -l` = 64 MiB. OverflowML 0.13.0 (`565dff8`). The GPU is shared by training (DINOv2 / FRCNN), video ingest,
feature extraction, Model Lab, llama-server — and the owner's games (cs2 took ~10.5 GB, 70–95 % GPU for 30 min).

## 1. Confirmed bugs (with repro)

### 1.1 Split GGUF planned from the first shard only — dangerous
```
overflowml llamacpp /mnt/w/Ai-Workstation/models/MiniMax-M2.5-GGUF/UD-Q4_K_XL/MiniMax-M2.5-UD-Q4_K_XL-00001-of-00004.gguf --ctx 16384 --idle --json
→ {"mode": "full_gpu", "est_vram_gb": 4.375, "vram_full_gpu_gb": 2.56}
```
The four shards are **123 GB**. The same applies to Qwen3.5-397B-A17B (5 shards, 177 GB), Step-3.5, Nemotron.
Fix: detect `-NNNNN-of-MMMMM.gguf`, read `split.count` / the tensor tables of every shard, sum sizes; refuse if a
shard is missing; test with a synthetic 3-shard fixture.

### 1.2 `measure_vram_headroom()` ignores other processes
It computes `total − own allocations`. While a game held 12 GB it still reported 31.8 GB free. Use the device's free
memory (`torch.cuda.mem_get_info()` or the nvidia-smi snapshot `detect --live` already has).

### 1.3 Batch sizing overshoots and WSL/Windows spills silently
`calculate_batch_size` from the batch-1 spike or a linear marginal: FRCNN got batch 58 (**0.44× speed**, 15.9 GB) and
batch 83 (31.6 GB → Windows shared-memory fallback: **no OOM, a silent crawl**). Per-item memory is not linear (FRCNN
batch 16 = 6.3 GB, batch 32 = 12.5 GB). Throughput knees measured: DINOv2-S batch 64 (1024 is only +1 % at 15× memory),
FRCNN batch 4. Fix: fit memory at two batch sizes, cap at the throughput knee (smallest batch within ~3 % of best),
never above `free − reserve`, and on WSL/Windows warn that exceeding VRAM does not raise.

### 1.4 Estimates too low (overhead)
Real VRAM exceeded `plan_gguf` by 0.37–0.75 GB (full GPU) and 1.2–2.0 GB (dense partial offload), `OVERHEAD_GB = 0.5`.
Suggest ≈1.2 GB full GPU, +1.0 GB when layers are split — better: learn it from the first launch (§3.3).

### 1.5 Thread rule reads the load average
`max(4, physical × (1 − load))` picked `-t 4`; a decode burst inflates the average. Measured on 35B-A3B `--n-cpu-moe 8`:
`-t 4` 123.4, `-t 8` **128.8**, default 15 → 122.6, `-t 30` 92.2 idle and **2.6 tok/s** next to a vision job.
Suggest `-t min(8, physical // 2)`, never above physical cores.

### 1.6 Dense partial offload is not a middle ground
27B `-ngl` 65 → 58: 60.9 → 12.8 tok/s (share 0.85 → 21 % speed); Mistral 41 → 38: 76.6 → 22.3. `LOW_GPU_SHARE = 0.5`
is far too permissive: for dense models warn/refuse below ~0.97 share and prefer "wait for VRAM" or an MoE model
(35B-A3B with `--n-cpu-moe 8` at ~16.8 GB gives 123 tok/s vs 7.5 for the 27B at `-ngl 52`).

## 2. Launch scripts found on the PC (owner's own, `/mnt/w/Ai-Workstation/scripts/launch-*.sh`)
`-ngl 99` on MoE models that can never fit 32 GB, `--mlock` on 100–177 GB models with a 64 MiB memlock limit,
`--host 0.0.0.0` **without an API key while WSL is in mirrored networking** (LAN-exposed), bare `-fa` (this build
requires `-fa on|off|auto`), no `LD_LIBRARY_PATH=/usr/lib/wsl/drivers/<nv_dispi…>` (CUDA aborts with
`free(): invalid pointer` and the wrapper may still exit 0). OverflowML's server launcher should make all of these
impossible by default (§3.4).

## 3. Product requirements (general, not MedSim-specific)

### 3.1 GPU gate as an open, documented standard
One lock + ledger any tool can join (MedSim, ComfyUI, training scripts, vLLM/Ollama launchers):
- Directory: default `~/.cache/overflowml/gpu_gate`, overridable with `OVERFLOWML_GATE_DIR`
  (MedSim today: `~/.cache/medsim/gpu_gate` — keep this format so both interoperate).
- `launch.lock`: exclusive advisory lock held while deciding + launching (flock on Linux/macOS, `msvcrt.locking` on
  Windows).
- `reservations.json`: `[{"job", "pid", "peak_gb", "launched_at", "settle_until"}]`; a reservation counts until
  `settle_until` (default launch + 20 s) while its pid is alive; stale entries (dead pid / expired) are dropped on read.
- Decision: `free_gb (device, from nvidia-smi) − reserve (1.5) − Σ pending reservations ≥ peak × 1.1` → launch;
  else wait (poll) or refuse **with the reason** (who holds what).
- One launch per snapshot: after launching, keep the lock until the settle window passes (or rely on the
  reservation) so the next decision sees the new job's memory.
- Reference implementation to mirror: MedSim `engine/gpu/overflow.py` (`LaunchGate`, `live_snapshot`, `gate_decision`).

### 3.2 `overflowml run` — the gate for ANY GPU command
```
overflowml run --name train-s2 --peak 13 [--wait 600] [--settle 20] -- python train.py …
```
Our worst problems were training/ingest jobs racing each other, not LLMs. `run` takes the lock, checks the live
snapshot, writes the reservation, execs the command (no shell string), streams its exit code, and explains waits /
refusals. Optional `--peak auto` from a learned profile (§3.3).

### 3.3 Measurement-driven planning
- `overflowml bench llamacpp <gguf> --sweep n-cpu-moe=0,2,4,6,8,10,14 [--ngl …] [--with-load "<cmd>"]` wrapping
  llama-bench (pp512/tg128), idle and next to a background GPU job; store the profile (VRAM, tok/s per placement) in
  the strategy cache; the planner then prefers measured configs over estimates.
- Learn real peak VRAM per model/ctx/placement (and per `run --name`) from the first launch and correct later plans.
- Our measured table (tok/s, tg128; idle / next to a GPU-bound DINOv2 job): 35B-A3B full GPU f16 KV 245 / 87
  (20.3 GB); q8_0 KV 235 / 85; `--n-cpu-moe` 2/4/6/8/10/14 → 188/150/134/123/106/84 idle, 62/49/40/34/30/24 loaded;
  27B full 60.9 / 26.1 (22.3 GB); Mistral-24B full 76.6 / 35.3 (20.9 GB). The planner's full-GPU pick was the fastest
  fitting config in every case. Mixed load hurts both sides: LLM −54…−64 %, the vision job −29…−44 % →
  **scheduling beats offloading**; expose time windows / priorities ("yield to training") as policy.

### 3.4 Safe server supervision defaults
Bind 127.0.0.1, API key file (0600) required for anything else, argv lists not shell strings, WSL
`LD_LIBRARY_PATH` fix applied automatically, detect the WSL `free(): invalid pointer` abort even when exit code is 0
(log scan + health probe), refuse `--mlock` when the model > memlock limit, warn on `networkingMode=mirrored` + non-loopback
bind, `-fa on` (never bare), refuse models bigger than RAM + VRAM (397B at 177 GB vs 128 GB RAM).

### 3.5 Other findings worth encoding
- Video decode inside a GPU process: **threads cost 3.7×** (GIL; 286 vs 1,047 frames/s); use decoder processes +
  shared memory. Splitting one video across 2–16 ffmpeg workers was slower; two videos at once +15 %. NVDEC in WSL2
  was 3× slower and not bit-identical.
- Precision gates per task: bf16 failed cosine ≥ 0.999 for DINOv2 features (0.9988) and matched only 91 % of FRCNN
  boxes (fp16 98.6 %); fp16 weights for DINOv2 passed (0.99998, 3.4×). Don't pick precision without a task check.
- Batch size changes results bitwise (max 0.031) → include it in any cache key.
- torch.compile: +15 %, 20 s warm-up → break-even ~420k frames.

## 4. Suggested PR order
1. Split-GGUF sizing + GGUF bounds hardening (1.1).
2. Gate standard (3.1) + `overflowml run` (3.2), compatible with MedSim's format.
3. Measurement fixes (1.2–1.6).
4. Model registry + `llm doctor`.
5. Admission rules with reasons → 0.14.0.
6. start/stop/status with safe defaults (3.4).
7. Supervision, yield-to-training, `overflowml bench` profiles, learned VRAM → 0.15.0.
8. Windows `ai-serve` wrapper, TUI screen, discovery; optional localhost OpenAI-compatible router.

Acceptance examples: MiniMax 4-shard plans ≥ 120 GB and never `full_gpu`; two `overflowml run` calls started 1 s
apart never both launch from the same snapshot; a game holding 12 GB is visible to batch sizing; FRCNN batch sizing
returns ≤ the knee; a server started with default flags is not reachable from the LAN.

## 5. Coordination with MedSimAI
MedSim keeps its own llama-server profiles (ports 8093–8097), its LLM manager (quality gates, licence register,
owner approvals, time windows) and its pinned llama.cpp build (`~/src/llama.cpp` 0.5.0 with the WSL fix) until a new
build is validated. MedSim will read `OVERFLOWML_GATE_DIR` and switch to the OverflowML gate once §3.1 ships;
until then OverflowML launchers should use `~/.cache/medsim/gpu_gate` with the format above so reservations are shared.
