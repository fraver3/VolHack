# VolHack AI working agreement

This repository is a team workspace for the EPFL VolHack Kaggle competition.

## Competition contract

Estimate `p_up`: the probability that the synthetic asset's price at minute 30 is higher than its price at minute 0. The binary target is `1` for higher and `0` for lower; exact ties were excluded.

- The data covers ten years of one-minute returns for one synthetic asset.
- Prediction cases are spaced three to seven hours apart and span roughly two hours. For each case, returns for minutes 0–15 are supplied; the remainder of the interval is withheld.
- The prediction compares minute 30 with minute 0. There are 15 unobserved minutes between the end of the observed window and the target time.
- All supplied observations are permissible, including those after a withheld interval. Returns spanning a withheld stretch are absent, so the move across that interval cannot be reconstructed.
- Submit exactly one finite probability in `[0, 1]` per supplied `ID`. `ID` is the Unix UTC timestamp of the target (minute 30); the required CSV columns are `ID,p_up`.
- Kaggle scores submissions with binary log loss. Lower is better, and confidently wrong probabilities are penalised heavily. Prioritize calibrated probabilities and protect validation from feature leakage.

Treat this section as the source of truth when planning features, data splits, calibration, and submission checks.

## Non-negotiable data protection

- `epfl-vol-hack/dataset.csv` and `epfl-vol-hack/sample_submission.csv` are supplied local assets. Treat both as **read-only**.
- Never edit, rename, delete, stage, force-add, commit, upload, or regenerate either file. Do not write beside them.
- Read them only through an explicit path. Generated artefacts belong under `outputs/`: models, cached features, and intermediate predictions stay ignored, while small validated CSVs in `outputs/submissions/` are versioned.
- The pre-commit hook rejects these data files. Keep it enabled with `git config core.hooksPath .githooks`.

## Working conventions

- Use the single project environment at `.venv`; do not create `venv`, `env`, Conda, or per-notebook environments in this repository.
- Put reusable Python in `src/volhack/`, runnable commands in `scripts/`, and exploratory notebooks in `notebooks/`.
- Before starting a modelling change, add an entry to `docs/EXPERIMENTS.md`; record the final configuration, validation scheme, log loss, and outcome when finished.
- Capture hypotheses and decisions in `docs/IDEAS.md`. Do not silently overwrite another contributor's note.
- Use time-aware validation: never let training features reveal observations unavailable at the prediction timestamp.
- Keep predictions finite and clipped strictly inside `(0, 1)` for log-loss stability; validate the `ID,p_up` submission schema locally.
- Run formatting and tests before proposing a commit. Do not create commits or push branches unless the user explicitly asks.

## Git conventions

- Work on `leo` by default. Before beginning work, run `git switch leo` and confirm `git branch --show-current` reports `leo`.
- Create a topic branch only when a task needs isolation, review, or concurrent work; merge or hand off the result back to `leo` deliberately.
- Work on a short-lived `feat/<name>-<scope>`, `fix/<name>-<scope>`, or `exp/<name>-<hypothesis>` branch.
- `leo` is the current integration branch. Rebase or merge deliberately; never rewrite shared history.
- Make focused commits that contain code and its experiment-log update. Include a small validated `outputs/submissions/*.csv` when it is the intended Kaggle submission; never commit supplied data, models, caches, or intermediate predictions.
