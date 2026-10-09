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

### 2026-10-09 — Leo — EXP-003 initial hypotheses

- Denser non-overlapping labels may reduce the variance caused by training on
  only about 9,700 of the available historical outcomes.
- Return volatility is persistent despite weak linear return autocorrelation.
  Normalize visible momentum by local volatility and test focused volatility
  context separately from the broad context ablation rejected in EXP-002.
- Use multiple chronological selection periods and conservative calibration;
  reserve the late-period check until the configuration is fixed.

### 2026-10-09 — Leo — EXP-003 selection decisions

- **Accepted:** dense training windows and normalized momentum. Chronological
  selection losses improve in all three periods; overlapping training labels
  are allowed within a split, while evaluation labels remain non-overlapping.
- **Accepted:** focused legal context and robust volatility statistics. The
  previous context rejection applied to the small EXP-002 training set. With
  dense training, enriched context reaches 0.472719 pooled selection loss in a
  fixed 50/50 blend of two-minute and five-minute models.
- **Rejected:** additional Platt calibration for this blend. Forward checks
  did not consistently improve log loss.
- **Accepted:** audit context completeness against real submission cases.
  The selected blend also beats the sparse baseline on the matched subset,
  although that subset is small and has higher absolute loss.
- **Open:** independently establish whether `ret` uses additive/logarithmic or
  simple-return semantics. Existing additive labels and compounded labels
  disagree in only 0.0716% of eligible windows; preserve the established
  convention for this final search rather than silently changing the target.

| Date | Owner | Decision | Reason |
| --- | --- | --- | --- |
| 2026-10-09 | Leo | Keep Kaggle source files local and read-only; version code, documentation, and small metadata only. | Prevent large data from polluting Git and preserve the supplied ground truth inputs. |
| 2026-10-09 | Leo | Version small, validated CSV submissions in `outputs/submissions/`; keep models, caches, predictions, and supplied data local. | Submission CSVs are compact reviewable competition artefacts, while other generated assets are not. |

### 2026-10-09 — Leo — EXP-003 final decision

**Accepted:** use `exp003_final_submission.csv` as the recommended final submission.
The fixed blend scored 0.468722 on the final chronological period,
versus 0.485108 for the rerun sparse baseline. The model
was selected before this check and then fitted on all eligible history.

### 2026-10-09 — Leo — EXP-004 initial distribution hypotheses

- Estimate `P(hidden_15_minute_move > -visible_move | legal observations)`
  directly. A positive typical return with rare large negative shocks can have
  nearly zero mean while producing materially more than 50% positive outcomes.
- Signed quantiles, robust core location, and positive/negative jump frequencies
  may identify distribution state more reliably than variance and skew moments.
- Test a conditional distribution estimator and a classifier with explicit
  shape features; combine only if chronological validation supports it.

### 2026-10-09 — Leo — EXP-004 selection decisions

- **Accepted:** model signed central mass and negative jumps separately. Returns
  have positive median and near-zero mean because negative shocks are larger.
- **Accepted:** add a monotone conditional survival estimator as a 10% component,
  together with 25% signed-tail classifier and 65% EXP-003. The fixed mixture
  improves all three selection folds and the context-matched subset.
- **Rejected:** iid local empirical convolution as a final probability model.
  Chronological losses around 0.52 trail the learned conditional models.
- **Accepted:** the fixed blend also improves the late-period check; the gain
  remains modest. Retain the EXP-003 submission for direct comparison.

### 2026-10-09 — Leo — EXP-004 final decision

**Accepted:** recommend `exp004_distribution_submission.csv`. The fixed blend
reduced late-period log loss from 0.468722292 to
0.468398652. The gain is small and consistent with selection results;
preserve EXP-003 for comparison.
