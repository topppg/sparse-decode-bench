<!-- first committed in the project repository as ff017b4 at 2026-11-21T23:51:41+01:00 -->

## PF-E2E amendment 2 (2026-11-21): box D relaunched after a setup failure

Written before any arm D data. Seen at writing: arm D's run 20261121T221741Z ended `FAILED sha q38` at 22:46:37 UTC, after its builds and all eight GATE K lines had passed and before `MODEL OK` / `SETUP DONE`: the model download exited non-zero with the file at 90%, so the partial file was never renamed and the SHA-256 check found no model. No gate B, reference or server row ran. Arms A, B and C are unaffected.

- **Change:** the download is resumed by hand on the same box (same pinned URL); the file is renamed to its final name only after its SHA-256 matches the registered hash, and the run script checks it again. Arm D is relaunched once on the same box in a fresh run directory, every stage repeated (builds, gate K, gate B, references, rows).
- **Cap:** the relaunch's run timeout is 9000 s instead of 10800 s; the deadline rule (rows admitted only if their estimate fits) is unchanged.
- Only the relaunched run is arm D's data. If it fails again, arm D is reported as not run.
