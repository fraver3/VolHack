# EPFL VolHack

Team workspace for the [EPFL VolHack Kaggle competition](https://www.kaggle.com/competitions/epfl-vol-hack): estimate `p_up`, the probability that the synthetic asset's price at minute 30 exceeds its price at minute 0. The competition uses binary log loss, so calibrated probabilities matter.

## Repository map

```text
AGENTS.md              Rules for humans and AI coding tools
docs/IDEAS.md          Hypotheses, decisions, and open questions
docs/EXPERIMENTS.md    Reproducible experiment register
docs/FILE_CATALOG.md   What belongs in Git and what stays local
docs/TEAM_WORKFLOW.md  Branches, ownership, review, and handoff
notebooks/             Exploratory analysis (keep outputs small)
scripts/               Reproducible command-line entry points
src/volhack/           Reusable code
outputs/               Ignored predictions/models; validated submission CSVs are versioned
```

## Data contract

The supplied files stay local at their original paths:

- `epfl-vol-hack/dataset.csv` — minute returns
- `epfl-vol-hack/sample_submission.csv` — required prediction IDs and schema

They are intentionally ignored, permission-locked read-only, and rejected by the pre-commit hook. They must never be modified or committed. Keep derived artefacts in `outputs/`; models and intermediate predictions remain local, while small validated CSVs in `outputs/submissions/` are versioned.

## Local setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
git config core.hooksPath .githooks
```

Use this single `.venv` for scripts, tests, and notebooks. The complete competition contract—including the allowed use of observations after hidden intervals and the target-time definition—is maintained in [AGENTS.md](AGENTS.md) so that coding agents apply it consistently.

Work on `leo` by default. Create a topic branch such as `git switch -c exp/leo-rolling-baseline` only when the task needs isolation or review. See [the team workflow](docs/TEAM_WORKFLOW.md) before beginning shared work.

## Submission guardrails

Each generated `submission.csv` must contain exactly the `ID` values from the supplied sample file, once each, with a finite `p_up` in `[0, 1]`. Generate it under `outputs/submissions/`, validate the schema locally, and commit the small final CSV with its experiment record.
