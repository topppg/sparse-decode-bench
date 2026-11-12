<!-- first committed in the project repository as b4d68ee at 2026-11-11T16:16:20+01:00 -->

## PORT-MQ (2026-11-11): first MoE model, Qwen3-30B-A3B-Instruct-2507, speed and RULER at 128k (pre-registered before any PORT-MQ data; PORT-G, PORT-RQ and the PORT-GQ pre-registration seen)

**Question.** Every result so far is on dense 3B–8B models. Does page-sparse decode carry over to a popular MoE model with plain GQA attention (48 layers, 32 query / 4 KV heads, head dim 128, 128 experts, 8 active), in speed and in quality?

**Engine.** The public release build, e7059e52a, `llama-bench` and `llama-sparse-tail --gen`. The run stops if the commit differs. One CUDA GPU with ≥ 48 GB; `CMAKE_CUDA_ARCHITECTURES` from the GPU. The fork has never run an MoE model: a smoke step (dense vs quest:12288 at depth 65536, 2 repetitions) is recorded first. If quest is not ≥ 1.02× dense there, the run stops and the result is reported as "sparse path not engaged / not faster", no RULER.

**Model.** `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF`, file `Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf`, SHA-256 47514861adaa7ae7f32818fe54595c7cbed2753417ba78df53054fcf7b9dcd88 (checked; a mismatch stops the run). Native context 262144, no RoPE change. KV q8_0.

**Speed (S).** `llama-bench -fa 1 -p 0 -n 64 -d <depth> -ctk q8_0 -ctv q8_0`, dense then quest (`LLAMA_SPARSE_DECODE=<budget>`, `LLAMA_SPARSE_DECODE_MIN_CTX=0`). Depths and budgets as the 8B rows of PORT-G: 8192=2048, 16384=3072, 32768=6144, 65536=12288, 131072=27648. 10 repetitions at depth ≤ 32768, 5 above.

**Quality (Q).** RULER, all 13 tasks, 20 samples per task, seed 42, `max_seq_length` 131072, `base` template, RULER commit db43f92, tokenizer `Qwen/Qwen3-30B-A3B-Instruct-2507` (as PORT-RQ with this tokenizer). Configs: dense, quest:27648. Scoring `port_r_score.py`; mean over the 13 tasks; paired bootstrap, 10000 resamples. The token file's SHA-256 is compared with PORT-RQ's `rulerq.i32` (adcc34b0…); equal or not is reported, no gate.

**Gate (Q).** Δ = quest − dense ≥ −3.0 points, and the 95% lower bound ≥ −6.0 (as R1, R3, RQ). If the dense mean is below 30, the Δ is reported as weak evidence.

**Predictions.**
- S: quest at 131072 is 1.5–2.3× dense (point 1.9). Reason: with ~3B active parameters, the KV read at 128k is a larger share of the bytes read per token than on the 8B (about 2/3 against about 1/2), so cutting it to ~21% gains more. MoE expert kernels are less bandwidth-efficient than dense GEMV, which pulls toward the low end.
- Q: PASS, Δ in [−3, +1], about 65% confident. Multi-needle tasks are again where any loss sits.

**Consequence (fixed now).**
- If S and Q are in by the time the Reddit post goes out, the post adds the MoE row with its speed and its RULER Δ and gate verdict, pass or fail. The GPU is named as it is (not a consumer card if it isn't one).
- If they are not in, the post goes out without it and the result follows as an update comment and in the bench README.
- If the smoke step fails, the post says the fork has not been made to work on MoE models yet.
- Published either way.
