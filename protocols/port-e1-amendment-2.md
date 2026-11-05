<!-- first committed in the project repository as 2694980 at 2026-11-04T12:24:02+01:00 -->

## PORT-E1 amendment 2 (2026-11-04, before any E1c data; E1a, E1b and E2 results on Llama-3.2-3B at 8k seen)

E1c uses the f32 page cache (`LLAMA_SPARSE_PAGE_STATS=f32`) instead of full recompute. E2a showed it gives logits bit-identical to full recompute on 4 configs × 4 books, and it is about 8× faster per tail. Gates, model, books and budgets are unchanged.
