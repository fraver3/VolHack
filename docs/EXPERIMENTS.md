# Experiment register

Every row should point to the exact code/configuration that produced the result. Never commit generated predictions, models, or large intermediate files; commit small, schema-validated final CSV submissions under `outputs/submissions/` and record their checksum here.

| ID | Date | Owner | Branch | Hypothesis | Validation / embargo | Features & model | Log loss | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-000 | — | — | — | — | — | — | — | Planned | Reserved for the first calibrated baseline. |
| EXP-001 | 2026-10-09 | Leo | leo | Short-horizon return summaries contain enough signal to improve over a historical base-rate forecast. | Chronological 70% / 15% / 15% train/validation/test split over 9,703 five-hour synthetic cases; 30-minute label/feature-span embargo at split boundaries. | Constant prior; regularized logistic regression; shallow histogram gradient boosting (`100` iterations, `7` leaves, `50` minimum leaf size) on 15 observable returns and summaries. | Test: 0.686171 / 0.530383 / **0.503081** (prior / logistic / boosting) | Complete | Validation log loss: 0.677448 / 0.491100 / 0.474953. Notebook: `notebooks/02_baseline_models.ipynb`. Submission: `outputs/submissions/exp001_hgb_submission.csv` (SHA-256 `383ad3e0a08b3c7df6919099612a4edfdf330b2e31204ed232b38e8665ed1919`). Inputs are read-only; probabilities clipped to `[1e-6, 1 - 1e-6]`. |
| EXP-002 | 2026-10-09 | Leo | leo | A fast model sweep can find a more stable, calibrated estimator than the EXP-001 fixed shallow booster. | Chronological 70% / 15% / 15% split of 9,702 five-hour cases (6,791 / 1,455 / 1,456), with a 30-minute purge at both boundaries. Selection and shrinkage used validation only. | Six approaches: regularized logistic regression; 100-tree extra trees; shallow/full-context HGB; medium/full-context HGB; medium HGB using only the 15 visible returns plus six summaries; and a 50/50 medium-HGB/linear blend. Full-context variants use legal 60/240-minute pre-gap and 15/60-minute post-gap summaries, skipping the hidden `target-14` through `target+91` span. | Validation: 0.494262 / 0.479531 / 0.474922 / 0.474095 / **0.472710** / 0.477808; selected visible-only HGB test: 0.511472 | Complete | `hgb_visible_only` was the validation winner (Brier 0.153840, AUC 0.846838); validation selected no shrinkage (weight 1.00). Its test loss did not beat EXP-001's 0.503081, but its requested submission was fit on all 9,703 eligible historical cases and written to `outputs/submissions/exp002_visible_hgb_submission.csv` (SHA-256 `94fc42c8905a050d17a8157dd4fc15fe5e2f006bdfeea3f13a28ea8f14abb6e2`). Full context was worse than the visible-only ablation, and the late test-period loss indicates temporal non-stationarity. Reproduce with `python scripts/generate_exp002_visible_submission.py`. |

## Experiment checklist

- [ ] Verify the dataset and sample-submission files were only read.
- [ ] Write the feature availability time and all lookback windows.
- [ ] Use a chronological split with an appropriate purge/embargo around withheld intervals.
- [ ] Record the probability clipping and calibration method.
- [ ] Validate generated IDs and `p_up` bounds against the supplied sample file.
- [ ] Update the row above before merging work into `leo`.
