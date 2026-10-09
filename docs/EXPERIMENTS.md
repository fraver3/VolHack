# Experiment register

Every row should point to the exact code/configuration that produced the result. Never commit generated predictions, models, or large intermediate files; save their local path and checksum here if needed.

| ID | Date | Owner | Branch | Hypothesis | Validation / embargo | Features & model | Log loss | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-000 | — | — | — | — | — | — | — | Planned | Reserved for the first calibrated baseline. |
| EXP-001 | 2026-10-09 | Leo | leo | Short-horizon return summaries contain enough signal to improve over a historical base-rate forecast. | Chronological 70% / 15% / 15% train/validation/test split over 9,703 five-hour synthetic cases; 30-minute label/feature-span embargo at split boundaries. | Constant prior; regularized logistic regression; shallow histogram gradient boosting (`100` iterations, `7` leaves, `50` minimum leaf size) on 15 observable returns and summaries. | Test: 0.686171 / 0.530383 / **0.503081** (prior / logistic / boosting) | Complete | Validation log loss: 0.677448 / 0.491100 / 0.474953. Notebook: `notebooks/02_baseline_models.ipynb`. Submission: `outputs/submissions/exp001_hgb_submission.csv` (SHA-256 `383ad3e0a08b3c7df6919099612a4edfdf330b2e31204ed232b38e8665ed1919`). Inputs are read-only; probabilities clipped to `[1e-6, 1 - 1e-6]`. |

## Experiment checklist

- [ ] Verify the dataset and sample-submission files were only read.
- [ ] Write the feature availability time and all lookback windows.
- [ ] Use a chronological split with an appropriate purge/embargo around withheld intervals.
- [ ] Record the probability clipping and calibration method.
- [ ] Validate generated IDs and `p_up` bounds against the supplied sample file.
- [ ] Update the row above before merging work into `leo`.
