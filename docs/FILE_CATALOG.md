# File catalog

Keep this catalog current when the repository layout changes. It is deliberately a catalog of **roles**, not an inventory of data rows or generated artefacts.

| Location | Purpose | Git policy | Owner / update trigger |
| --- | --- | --- | --- |
| `AGENTS.md` | Shared instructions for humans and AI coding tools | Versioned | Update when safety or workflow rules change. |
| `README.md` | Project entry point and setup | Versioned | Update when setup or layout changes. |
| `docs/IDEAS.md` | Hypothesis and decision backlog | Versioned | Update for every material idea or decision. |
| `docs/EXPERIMENTS.md` | Experiment metadata and results | Versioned | Update for every modelling run worth comparing. |
| `docs/TEAM_WORKFLOW.md` | Collaboration and branch protocol | Versioned | Update when team process changes. |
| `src/volhack/` | Reusable, reviewed code | Versioned | Code owner for the relevant feature. |
| `scripts/` | Reproducible entry points | Versioned | Script author. |
| `notebooks/` | Compact exploratory work | Versioned | Notebook author; clear bulky outputs before commit. |
| `tests/` | Automated checks | Versioned | Change alongside behavior. |
| `epfl-vol-hack/dataset.csv` | Supplied return history | **Ignored and read-only** | Never modify, stage, or commit. |
| `epfl-vol-hack/sample_submission.csv` | Supplied IDs and schema | **Ignored and read-only** | Never modify, stage, or commit. |
| `epfl-vol-hack.zip` | Kaggle archive inherited from the initial repository history | Already versioned; leave untouched | Do not replace, recommit, or add copies without a team decision. |
| `outputs/predictions/`, `outputs/models/`, `outputs/cache/` | Predictions, models, and cached features | Ignored | Generated locally only. |
| `outputs/submissions/*.csv` | Small, schema-validated Kaggle submissions | Versioned | Submission author; commit with the matching experiment record and checksum. |

Before committing, run `git diff --cached --name-only` and compare the paths with this table. If supplied data, a model, a cache, or an intermediate prediction appears as staged, stop and remove it from the index rather than committing it.
