<!-- first committed in the project repository as 5932f27 at 2026-11-06T14:20:30+01:00 -->

## PORT-L amendment 2 (2026-11-06): plain quest in Phase 3 (written after PORT-D D-A was read; the first Phase 3 attempt with quest:d2 had run its speed bench at 131072 and part of 262144, not read; no L2/L3 data; D-B finished, not read)

Phase 3 is restarted with `quest` and no dense layers (`quest:49152`, `quest:98304`). Reason: quest:d2 won the PORT-D rule by 0.05 points (quest:d2 − quest = +0.04 [−3.12, 3.00]); the configuration that will be released is plain quest, and the clean feature branch has no dense-layer option. Measuring the released configuration takes priority over the literal tie-break. The bench files of the aborted attempt are kept in `bench/p3_try1_d2/` and reported as such. Everything else in amendment 1 stands, with quest in place of quest:d2.
