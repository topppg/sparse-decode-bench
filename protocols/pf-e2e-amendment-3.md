<!-- first committed in the project repository as d875a39 at 2026-11-22T00:00:21+01:00 -->

## PF-E2E amendment 3 (2026-11-21): box D's code from on-box mirrors

Written before any arm D data. Seen at writing: amendment 2's relaunch (run 20261121T225103Z) ended `FAILED clone` at 22:55:15 UTC (`curl 56 GnuTLS recv error`, `fetch-pack: unexpected disconnect`): the box's network fails on long transfers (the model download had stalled the same way). Nothing past PROMPTS OK ran.

- **Change:** the fork and upstream sources come from local bare mirrors on the box, made from the first run's clones (fetched from GitHub at 22:18–22:24 UTC and verified by that run: SOURCE OK tree 9e3e51c75, UPSTREAM OK c6a25fc62). Git's `url.<mirror>.insteadOf` maps the two GitHub URLs to them; the run script is unchanged and still checks the fork commit f7d80009d, the trees after both patch sets and the upstream commit, so a wrong source fails as before.
- **Relaunch:** run 20261121T225917Z at 22:59:17 UTC, run timeout 8400 s. It passed SOURCE OK and UPSTREAM OK.
- Amendment 2's last rule stands: only this run is arm D's data; another failure means arm D is reported as not run.
