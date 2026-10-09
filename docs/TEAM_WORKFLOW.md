# Team workflow

## Branch plan

`leo` is the default and current integration branch. Start ordinary work there. Create a short-lived, descriptive branch only when the work needs isolation, parallel development, or review:

| Work type | Pattern | Example |
| --- | --- | --- |
| Feature or infrastructure | `feat/<name>-<scope>` | `feat/maya-feature-pipeline` |
| Bug fix | `fix/<name>-<scope>` | `fix/noah-id-validation` |
| Isolated modelling trial | `exp/<name>-<hypothesis>` | `exp/leo-volatility-regime` |

When a topic branch is needed, create it from the current integration point and publish only the branch you own:

```sh
git switch leo
git pull --ff-only origin leo
git switch -c exp/<your-name>-<hypothesis>
```

If `leo` has not yet been published, use `master` as the pull source until its first push, then update this command. Never force-push `leo` or another contributor's branch.

## Shared ownership

- One person owns a branch at a time; coordinate before editing the same notebook, log row, or module.
- Every model change includes a `docs/EXPERIMENTS.md` update; every strategic conclusion updates `docs/IDEAS.md`.
- Prefer a small reusable module plus a compact notebook to a large notebook-only workflow.
- Review staged paths before every commit: `git diff --cached --name-only`.

## Data and artefacts

The two supplied CSV files are immutable local inputs. Do not commit them, copies of them, trained models, cached features, or generated submissions. Use `outputs/` for all derived files. The existing zip archive is inherited from the initial repository history; leave it untouched and do not add replacement archives.

The local pre-commit hook is enabled with:

```sh
git config core.hooksPath .githooks
```

It blocks attempts to stage `dataset.csv` or `sample_submission.csv`. The `.gitignore` is the first line of defense; the hook also catches `git add -f`.

## Handoff checklist

1. Rebase or merge the latest `leo` deliberately and resolve conflicts locally.
2. Run `ruff check .` and `pytest` when code changes.
3. Ensure the experiment and idea logs describe the result.
4. Confirm `git diff --cached --name-only` excludes protected data and `outputs/`.
5. Open a review or hand off the branch with the experiment ID and validation log loss.
