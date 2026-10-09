"""Run EXP-002's fast, leakage-audited chronological model sweep.

The context features deliberately skip the unavailable return span from target-14
through target+91.  Values after that span are allowed by the competition
contract and are represented only by summaries and their finite-observation
fractions; no missing return is reconstructed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from volhack.baselines import PROBABILITY_EPSILON, make_exp002_visible_hgb

RANDOM_SEED = 42
CASE_SPACING_MINUTES = 300
EMBARGO_MINUTES = 30


def _window_summaries(values: np.ndarray) -> np.ndarray:
    """Summarise return windows, retaining availability as a separate feature."""
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    safe = np.where(finite, values, 0.0)
    mean = np.divide(safe.sum(axis=1), count, out=np.zeros(len(values)), where=count > 0)
    centered = np.where(finite, values - mean[:, None], 0.0)
    std = np.sqrt(
        np.divide((centered**2).sum(axis=1), count, out=np.zeros(len(values)), where=count > 0)
    )
    return np.column_stack(
        [
            safe.sum(axis=1),
            mean,
            std,
            np.abs(safe).sum(axis=1),
            np.divide((safe > 0).sum(axis=1), count, out=np.zeros(len(values)), where=count > 0),
            count / values.shape[1],
        ]
    )


def _context_features(ret: np.ndarray, target_rows: np.ndarray) -> np.ndarray:
    """Return only data available outside each target's hidden 106-minute gap."""
    visible = ret[target_rows[:, None] - np.arange(29, 14, -1)]
    pre_60 = ret[target_rows[:, None] - np.arange(30, 90)]
    pre_240 = ret[target_rows[:, None] - np.arange(90, 330)]
    # For supplied cases, the hidden stretch ends at target + 91.
    post_15 = ret[target_rows[:, None] + np.arange(92, 107)]
    post_60 = ret[target_rows[:, None] + np.arange(92, 152)]
    observed = _window_summaries(visible)
    context = np.column_stack(
        [
            _window_summaries(pre_60),
            _window_summaries(pre_240),
            _window_summaries(post_15),
            _window_summaries(post_60),
        ]
    )
    return np.column_stack([visible, observed, context])


def make_cases(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create separated fully labelled historical cases with available prehistory."""
    ret = returns["ret"].to_numpy(dtype=float)
    candidates = np.arange(330, len(ret) - 152, CASE_SPACING_MINUTES)
    finite = np.isfinite(ret)
    finite_prefix = np.r_[0, np.cumsum(finite)]
    # Labels require all 30 returns from target-29 through target.  Feature
    # summaries tolerate missing values in their legal external context.
    complete_label = finite_prefix[candidates + 1] - finite_prefix[candidates - 29] == 30
    target_rows = candidates[complete_label]
    ret_prefix = np.r_[0.0, np.cumsum(np.nan_to_num(ret, nan=0.0))]
    moves = ret_prefix[target_rows + 1] - ret_prefix[target_rows - 29]
    non_ties = moves != 0.0
    target_rows = target_rows[non_ties]
    labels = (moves[non_ties] > 0.0).astype(int)
    return target_rows, _context_features(ret, target_rows), labels


def masks_for_chronological_split(
    target_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make a 70/15/15 split with a 30-minute purge at both boundaries."""
    validation_start = target_rows[int(len(target_rows) * 0.70)]
    test_start = target_rows[int(len(target_rows) * 0.85)]
    train = target_rows < validation_start - EMBARGO_MINUTES
    validation = (target_rows >= validation_start) & (target_rows < test_start - EMBARGO_MINUTES)
    test = target_rows >= test_start
    return train, validation, test


def score(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    return {
        "log_loss": log_loss(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
        "auc": roc_auc_score(y_true, probability),
    }


def models() -> dict[str, tuple[object, slice]]:
    """Six quick, intentionally conservative model families/configurations."""
    linear = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.03, max_iter=1_000, random_state=RANDOM_SEED),
    )
    return {
        "linear_regularized": (linear, slice(None)),
        "extra_trees_100": (
            make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesClassifier(
                    n_estimators=100,
                    max_features=0.7,
                    min_samples_leaf=20,
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
            slice(None),
        ),
        "hgb_shallow": (
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=7,
                min_samples_leaf=50,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=RANDOM_SEED,
            ),
            slice(None),
        ),
        "hgb_medium": (
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.04,
                max_leaf_nodes=15,
                min_samples_leaf=80,
                l2_regularization=3.0,
                early_stopping=False,
                random_state=RANDOM_SEED,
            ),
            slice(None),
        ),
        # Raw 15 observed returns plus their summaries: this is an ablation of
        # the legal pre-/post-gap context rather than an expanded model search.
        "hgb_visible_only": (make_exp002_visible_hgb(RANDOM_SEED), slice(0, 21)),
        "hgb_medium_blend_linear": (
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.04,
                max_leaf_nodes=15,
                min_samples_leaf=80,
                l2_regularization=3.0,
                early_stopping=False,
                random_state=RANDOM_SEED,
            ),
            slice(None),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("epfl-vol-hack/dataset.csv"))
    parser.add_argument(
        "--output-path", type=Path, default=Path("outputs/cache/exp002_sweep_metrics.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    returns = pd.read_csv(args.data_path)
    if returns.columns.tolist() != ["timestamp", "ret"]:
        raise ValueError("Expected exactly the columns ['timestamp', 'ret'].")
    if not np.all(np.diff(returns["timestamp"].to_numpy()) == 60):
        raise ValueError("Dataset timestamps must be on a regular one-minute grid.")

    target_rows, features, labels = make_cases(returns)
    train, validation, test = masks_for_chronological_split(target_rows)
    print(
        f"Cases: train={train.sum():,}, validation={validation.sum():,}, "
        f"test={test.sum():,}; features={features.shape[1]}"
    )

    results: list[dict[str, float | str]] = []
    fitted: dict[str, object] = {}
    feature_columns_by_model: dict[str, slice] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    for name, (model, feature_columns) in models().items():
        feature_columns_by_model[name] = feature_columns
        model.fit(features[train, feature_columns], labels[train])
        fitted[name] = model
        if name == "hgb_medium_blend_linear":
            probability = 0.5 * model.predict_proba(features[validation, feature_columns])[:, 1]
            probability += (
                0.5 * fitted["linear_regularized"].predict_proba(features[validation])[:, 1]
            )
        else:
            probability = model.predict_proba(features[validation, feature_columns])[:, 1]
        validation_probabilities[name] = probability
        results.append(
            {"model": name, "split": "validation", **score(labels[validation], probability)}
        )

    # Select shrinkage solely on the validation interval.  It protects log loss
    # from rare overconfident errors without fitting a flexible calibrator on a
    # relatively small calibration slice.
    training_prior = labels[train].mean()
    selected_model = min(
        validation_probabilities,
        key=lambda name: log_loss(labels[validation], validation_probabilities[name]),
    )
    selected_validation = validation_probabilities[selected_model]
    shrinkages = np.linspace(0.0, 1.0, 21)
    best_shrinkage = min(
        shrinkages,
        key=lambda weight: log_loss(
            labels[validation], weight * selected_validation + (1.0 - weight) * training_prior
        ),
    )
    selected_test = fitted[selected_model].predict_proba(
        features[test, feature_columns_by_model[selected_model]]
    )[:, 1]
    results.extend(
        [
            {"model": selected_model, "split": "test", **score(labels[test], selected_test)},
            {
                "model": f"{selected_model}_shrink_{best_shrinkage:.2f}",
                "split": "test",
                **score(
                    labels[test],
                    best_shrinkage * selected_test + (1.0 - best_shrinkage) * training_prior,
                ),
            },
        ]
    )
    table = pd.DataFrame(results).sort_values(["split", "log_loss"])
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Validation-selected model: {selected_model}")
    print(f"Validation-selected shrinkage weight: {best_shrinkage:.2f}")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
