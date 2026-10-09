# Ideas and decisions

Use this file as the shared hypothesis backlog. Add your initials, date, and links to the corresponding experiment entry. Mark an idea **accepted**, **rejected**, or **open** instead of deleting it.

| Status | Owner | Hypothesis / decision | Evidence / next action |
| --- | --- | --- | --- |
| Open | Team | Establish a chronological, purged validation scheme before comparing models. Random folds could leak surrounding return context. | Define the prediction timestamps and embargo window in the first baseline experiment. |
| Open | Team | Use the 15 observed minutes to derive cumulative return, volatility, sign/run statistics, and short-horizon momentum features. | Compare against a constant-probability calibration baseline. |
| Open | Team | Add time-of-day, day-of-week, and long-history regime features using only information available at each case. | Audit every feature's lookback and availability time. |
| Open | Team | Calibrate model scores on an untouched chronological validation slice because the leaderboard uses log loss. | Evaluate clipping and calibration separately from raw discrimination. |
| Accepted | Leo | Compare simple momentum/dispersion features with a constant prior before adding long-history or complex features. | EXP-001's held-out chronological test log loss is 0.503081 for shallow boosting, versus 0.686171 for the prior; retain this benchmark before extending features. |
| Accepted | Leo | Treat each submission target as a fixed position inside its corresponding missing-data window when building features and validation. | Initial audit confirms all 17,517 targets are 14 minutes after the start of a 106-minute missing-return run. Keep this alignment explicit and do not infer returns through the gap. |
| Accepted | Leo | Test several low-cost estimators with identical feature availability and a purged chronological split; favor consistency across validation and final test over a single favorable split. | EXP-002 compared linear, bagged trees, boosted trees, a feature ablation, and a blend; retain the test result rather than selecting on validation alone. |
| Rejected | Leo | Add broad legal pre-gap and post-gap return summaries to the 15 visible returns. | EXP-002: medium full-context HGB validation loss was 0.474095, versus 0.472710 for the otherwise identical visible-only model; do not extend this context feature set without a new mechanism. |
| Accepted | Leo | Keep EXP-001's shallow HGB as the current submission benchmark. | EXP-002's selected medium visible-only HGB reached 0.472710 on validation but 0.511472 on test, worse than EXP-001's 0.503081. The faster linear, extra-trees, and blend variants also trailed on validation. |

## Decision log

| Date | Owner | Decision | Reason |
| --- | --- | --- | --- |
| 2026-10-09 | Leo | Keep Kaggle source files local and read-only; version code, documentation, and small metadata only. | Prevent large data from polluting Git and preserve the supplied ground truth inputs. |
| 2026-10-09 | Leo | Version small, validated CSV submissions in `outputs/submissions/`; keep models, caches, predictions, and supplied data local. | Submission CSVs are compact reviewable competition artefacts, while other generated assets are not. |
