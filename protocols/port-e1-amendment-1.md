<!-- first committed in the project repository as a9ec3c7 at 2026-11-04T11:18:29+01:00 -->

## PORT-E1 amendment 1 (2026-11-04, before any E1 data on the protocol model; development runs on Qwen2.5-1.5B q8_0 seen)

Seen: on Qwen2.5-1.5B (not the protocol model) the E1a check found 0 differing pages in 32/32 cases and 0 differing mask cells in 8/8; page kernel vs token kernel mean KL 1.6–8.8e-6 per book (4 books, B = 2048, T = 8192).

Change: E1a–E1c use q8_0 weights (KV cache stays f16). Loading the f16 3B GGUF was killed twice by the memory guard (free 36% → 20% and 53% → 18%; the Mac already had ~20 GB of swap in use). If memory is freed before E1c runs, E1c is run with both f16 and q8_0 weights and both are reported; the gate applies to f16.
