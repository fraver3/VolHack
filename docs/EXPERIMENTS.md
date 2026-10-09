# Experiment register

Every row should point to the exact code/configuration that produced the result. Never commit generated predictions, models, or large intermediate files; save their local path and checksum here if needed.

| ID | Date | Owner | Branch | Hypothesis | Validation / embargo | Features & model | Log loss | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-000 | — | — | — | — | — | — | — | Planned | Reserved for the first calibrated baseline. |

## Experiment checklist

- [ ] Verify the dataset and sample-submission files were only read.
- [ ] Write the feature availability time and all lookback windows.
- [ ] Use a chronological split with an appropriate purge/embargo around withheld intervals.
- [ ] Record the probability clipping and calibration method.
- [ ] Validate generated IDs and `p_up` bounds against the supplied sample file.
- [ ] Update the row above before merging work into `leo`.
