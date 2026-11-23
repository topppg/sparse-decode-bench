<!-- first committed in the project repository as d7bf656 at 2026-11-21T23:22:56+01:00 -->

## PF-E2E amendment 1 (2026-11-21): RAM floor for the 5090 boxes

Written before any PF-E2E data: box D (RTX 4090) had passed gate P and was building; no gate B, reference or timed row existed on any box.

- **Seen:** the three rented RTX 5090 boxes (vastai48, vastai49, vastai50) report 30 GiB of host RAM (MemTotal ~31.9 GB), below gate P's registered ≥ 60 GiB for arms A, B and C. Rented 5090s with ≥ 64 GB of host RAM are rare.
- **Change:** gate P's RAM floor for arms A, B and C becomes ≥ 28 GiB (`run_pfe_speed.sh` arm A `RAM_MIN=28`, `run_pfe_ruler.sh`). Arm D's ≥ 30 GiB, every other part of gate P, the matrix, gates and decisions are unchanged.
- **Why it does not change the measurement:** the model and the KV cache are on the GPU (`-ngl 99`); host RAM holds the page cache of the 16.6 GB model file and the client. Earlier 5090 runs of the same model on 30 GiB boxes completed without memory trouble. Build concurrency is already bounded by MemAvailable / 2 GiB (recorded as BUILD JOBS).
- **Watch:** host memory (`free`, the kernel log for OOM kills) is checked during each 5090 box's first build and first 256k row; an OOM-killed process is recorded and handled by the existing rules.
- Box D runs the pre-amendment copy of `run_pfe_speed.sh`; it differs from the amended file only in arm A's constant and the header comment.
