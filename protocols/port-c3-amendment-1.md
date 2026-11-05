<!-- first committed in the project repository as 4cdf06c at 2026-11-05T17:10:51+01:00 -->

## PORT-C3 amendment 1 (2026-11-05): the 8B on the fixed build (written before any 8B timing on it; all PORT-C3 3B results seen)

Same build (d96e8140f), box `vastai8`, `llama-bench` exactly as PORT-C C3 for the 8B: Llama-3.1-8B q8_0, KV q8_0, depths 8k / 32k / 128k, budgets 2048 / 6144 / 27648, dense and sparse in the same run. The 8B q8_0 file is rebuilt on this box. Its SHA-256 is compared with the PORT-C file from `vastai7` (38a8de72…), and a mismatch is reported.

Prediction, from the old C3 8B times and the 3B savings scaled by 32 layers and 32 heads: S(8B, 128k) = 1.30, band [1.15, 1.45]; S(32k) in [0.9, 1.1]; S(8k) in [0.75, 0.95]. Usefulness bar, unchanged from PORT-C: S(8B, 128k) ≥ 1.2. This adds measurements only; the engineering phase stays closed.
