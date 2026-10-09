# Experiment register

Every row should point to the exact code/configuration that produced the result. Never commit generated predictions, models, or large intermediate files; commit small, schema-validated final CSV submissions under `outputs/submissions/` and record their checksum here.

| ID | Date | Owner | Branch | Hypothesis | Validation / embargo | Features & model | Log loss | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-000 | — | — | — | — | — | — | — | Planned | Reserved for the first calibrated baseline. |
| EXP-001 | 2026-10-09 | Leo | leo | Short-horizon return summaries contain enough signal to improve over a historical base-rate forecast. | Chronological 70% / 15% / 15% train/validation/test split over 9,703 five-hour synthetic cases; 30-minute label/feature-span embargo at split boundaries. | Constant prior; regularized logistic regression; shallow histogram gradient boosting (`100` iterations, `7` leaves, `50` minimum leaf size) on 15 observable returns and summaries. | Test: 0.686171 / 0.530383 / **0.503081** (prior / logistic / boosting) | Complete | Validation log loss: 0.677448 / 0.491100 / 0.474953. Notebook: `notebooks/02_baseline_models.ipynb`. Submission: `outputs/submissions/exp001_hgb_submission.csv` (SHA-256 `383ad3e0a08b3c7df6919099612a4edfdf330b2e31204ed232b38e8665ed1919`). Inputs are read-only; probabilities clipped to `[1e-6, 1 - 1e-6]`. |
| EXP-002 | 2026-10-09 | Leo | leo | A fast model sweep can find a more stable, calibrated estimator than the EXP-001 fixed shallow booster. | Chronological 70% / 15% / 15% split of 9,702 five-hour cases (6,791 / 1,455 / 1,456), with a 30-minute purge at both boundaries. Selection and shrinkage used validation only. | Six approaches: regularized logistic regression; 100-tree extra trees; shallow/full-context HGB; medium/full-context HGB; medium HGB using only the 15 visible returns plus six summaries; and a 50/50 medium-HGB/linear blend. Full-context variants use legal 60/240-minute pre-gap and 15/60-minute post-gap summaries, skipping the hidden `target-14` through `target+91` span. | Validation: 0.494262 / 0.479531 / 0.474922 / 0.474095 / **0.472710** / 0.477808; selected visible-only HGB test: 0.511472 | Complete | `hgb_visible_only` was the validation winner (Brier 0.153840, AUC 0.846838); validation selected no shrinkage (weight 1.00). Its test loss did not beat EXP-001's 0.503081, but its requested submission was fit on all 9,703 eligible historical cases and written to `outputs/submissions/exp002_visible_hgb_submission.csv` (SHA-256 `94fc42c8905a050d17a8157dd4fc15fe5e2f006bdfeea3f13a28ea8f14abb6e2`). Full context was worse than the visible-only ablation, and the late test-period loss indicates temporal non-stationarity. Reproduce with `python scripts/generate_exp002_visible_submission.py`. |
| EXP-003 | 2026-10-09 | Leo | leo | Dense training and normalized momentum with robust legal volatility context improve stability. | Three expanding chronological folds; feature-span purge; final 15% check after selection. | 50/50 enriched HGB blend; exact settings in `docs/exp003_final_selection.json`. | Selection: **0.472719**; late test: **0.468722** versus 0.485108 sparse baseline | Complete | 17,517-row `outputs/submissions/exp003_final_submission.csv`; SHA-256 `73de0041a5555964f225fbd9b2492234d115cbbf28babe12dc6a43203c9e16e5`. No additional calibration. Four tests and formatting pass. |

## EXP-003 — final submission search (2026-10-09)

- Owner/branch: Leo / `leo`; 20-minute search budget.
- Hypothesis: using non-overlapping 30-minute training cases rather than five-hour
  subsampling, together with volatility-normalized visible momentum and legal
  context volatility, improves stability over EXP-001/002.
- Planned selection: expanding chronological folds ending at 55%, 70%, and 85%
  of the timeline, with purging for the complete feature/label span; final 15%
  assessed once after model and calibration selection. Previous experiments have
  already reported aggregate results on this late period, so it is not a wholly
  new competition holdout.
- Feature audit: visible returns end at target-15. External context skips the
  entire target-14 through target+91 missing span. Any post-gap features use only
  supplied observations as explicitly permitted by the competition contract.
- Selection complete: 50/50 `dense2_enriched` / `dense5_enriched_deeper`.
  Selection fold losses: 0.477982 / 0.472929 / 0.467222; pooled loss 0.472719,
  versus 0.491191 for the same-fold sparse baseline. Late-period evaluation and
  final full-history fits are complete; final scores and checksum follow below.
- Configuration: enriched features (113 columns); two-minute training stride,
  15 leaves / 500 iterations / 750 minimum leaf samples / L2=30; five-minute
  stride, 31 leaves / 350 iterations / 500 minimum leaf samples / L2=20.
  Both use learning rate 0.05, seed 42, and no internal early stopping.
- Availability audit refinement: enriched context reaches target-749 through
  target-30 and target+92 through target+211. The entire target-14 through
  target+91 span is excluded. Array boundaries are masked, never wrapped.
  Training feature spans end before the earliest validation feature span.
- Calibration: forward Platt checks gave 0.47299998 versus raw 0.47292929 on
  fold 1, and 0.46720864 versus raw 0.46722194 on fold 2. Improvement was not
  consistent, so no additional calibration was retained. Clip at 1e-6.
- Sampling audit: actual submission pre-60 context is 99.992% observed and
  post-60 context is entirely observed. On the 3,186 selection cases matching
  those completeness constraints, the selected blend scored 0.497526 versus
  0.514171 for the sparse baseline. Historical context availability differs
  from submission availability, so pooled scores are not guaranteed Kaggle
  loss estimates.
- Label convention: preserve EXP-001/002's additive 30-minute return definition.
  An audit found compound/simple-return signs differ on 0.0716% of eligible
  30-minute cases. Return semantics are not independently documented in the
  supplied two-column CSV; this is a small residual target-definition caveat.
- Reproduce the final fixed model: `.venv/bin/python
  scripts/generate_exp003_submission.py`. Configuration is versioned in
  `docs/exp003_final_selection.json`; feature caches are optional and ignored.
- Reproduce selection: run `scripts/run_exp003_search.py`,
  `scripts/run_exp003_refinement.py`, `scripts/run_exp003_density.py`, then
  `scripts/select_exp003_final.py`, using `.venv/bin/python` for each.
- Runtime versions: Python 3.9.6, NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1,
  SciPy 1.13.1, threadpoolctl 3.7.0.

- Final late-period check: **0.468722** log loss versus
  0.485108 for the rerun sparse baseline, on 14,437
  non-overlapping cases. Brier 0.153784; AUC 0.848045.
  Context-matched late subset: 1,024 cases, selected loss
  0.487633 versus baseline 0.506983.
- Final training: 1,445,290 two-minute windows and 578,115 five-minute windows.
- Final submission: `outputs/submissions/exp003_final_submission.csv`; 17,517
  IDs in exact supplied order, finite probabilities in
  [0.001119965, 0.996010071].
  SHA-256 `73de0041a5555964f225fbd9b2492234d115cbbf28babe12dc6a43203c9e16e5`.
- Validation complete: Ruff lint/format checks pass; four tests pass (hidden-span
  invariance, boundary masking, complete label windows, submission validation).
  Both supplied-file SHA-256 hashes match their initial values. Checks completed
  before committing; no push or Kaggle upload was performed. Status: complete.

## Experiment checklist

- [ ] Verify the dataset and sample-submission files were only read.
- [ ] Write the feature availability time and all lookback windows.
- [ ] Use a chronological split with an appropriate purge/embargo around withheld intervals.
- [ ] Record the probability clipping and calibration method.
- [ ] Validate generated IDs and `p_up` bounds against the supplied sample file.
- [ ] Update the row above before merging work into `leo`.
