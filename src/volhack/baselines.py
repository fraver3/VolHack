"""Feature construction and baseline modelling for the VolHack competition."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

VISIBLE_OFFSETS = np.arange(29, 14, -1)
PROBABILITY_EPSILON = 1e-6


def make_feature_matrix(returns: np.ndarray, target_rows: np.ndarray) -> np.ndarray:
    """Build features using only returns from target-29 through target-15.

    The competition target is the price at ``target`` relative to 30 minutes
    earlier. These 15 returns are the observable portion of that interval.
    """
    returns = np.asarray(returns, dtype=float)
    target_rows = np.asarray(target_rows, dtype=int)
    visible_rows = target_rows[:, None] - VISIBLE_OFFSETS

    if visible_rows.min() < 0 or visible_rows.max() >= len(returns):
        raise ValueError("At least one target does not have a complete visible window.")

    visible_returns = returns[visible_rows]
    if not np.isfinite(visible_returns).all():
        raise ValueError("At least one target has a missing return in its visible window.")

    sign_changes = (np.diff(np.signbit(visible_returns), axis=1) != 0).sum(axis=1)
    return np.column_stack(
        [
            visible_returns,
            visible_returns.sum(axis=1),
            visible_returns.mean(axis=1),
            visible_returns.std(axis=1),
            np.abs(visible_returns).sum(axis=1),
            (visible_returns > 0).mean(axis=1),
            visible_returns[:, -3:].sum(axis=1),
            visible_returns[:, -5:].sum(axis=1),
            visible_returns[:, -10:].sum(axis=1),
            visible_returns.min(axis=1),
            visible_returns.max(axis=1),
            sign_changes,
        ]
    )


def make_exp002_visible_features(returns: np.ndarray, target_rows: np.ndarray) -> np.ndarray:
    """Build EXP-002's raw visible-return and six summary features.

    This intentionally excludes the broader pre-/post-gap context tested in
    EXP-002, whose ablation did not improve validation performance.
    """
    returns = np.asarray(returns, dtype=float)
    target_rows = np.asarray(target_rows, dtype=int)
    visible_rows = target_rows[:, None] - VISIBLE_OFFSETS

    if visible_rows.min() < 0 or visible_rows.max() >= len(returns):
        raise ValueError("At least one target does not have a complete visible window.")

    visible_returns = returns[visible_rows]
    if not np.isfinite(visible_returns).all():
        raise ValueError("At least one target has a missing return in its visible window.")

    return np.column_stack(
        [
            visible_returns,
            visible_returns.sum(axis=1),
            visible_returns.mean(axis=1),
            visible_returns.std(axis=1),
            np.abs(visible_returns).sum(axis=1),
            (visible_returns > 0).mean(axis=1),
            np.ones(len(visible_returns)),
        ]
    )


def make_historical_cases(
    returns: pd.DataFrame, case_spacing_minutes: int = 300
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create fully observed, non-overlapping-ish historical training cases."""
    if returns.columns.tolist() != ["timestamp", "ret"]:
        raise ValueError("Expected exactly the columns ['timestamp', 'ret'].")
    if case_spacing_minutes < 30:
        raise ValueError("Case spacing must be at least the 30-minute label span.")

    ret = returns["ret"].to_numpy(dtype=float)
    observed = np.isfinite(ret)
    candidate_rows = np.arange(30, len(returns), case_spacing_minutes)
    observed_prefix = np.r_[0, np.cumsum(observed)]
    complete_windows = (
        observed_prefix[candidate_rows + 1] - observed_prefix[candidate_rows - 29] == 30
    )
    target_rows = candidate_rows[complete_windows]

    return_prefix = np.r_[0.0, np.cumsum(np.nan_to_num(ret, nan=0.0))]
    thirty_minute_moves = return_prefix[target_rows + 1] - return_prefix[target_rows - 29]
    non_ties = thirty_minute_moves != 0
    target_rows = target_rows[non_ties]
    labels = (thirty_minute_moves[non_ties] > 0).astype(int)

    return target_rows, make_feature_matrix(ret, target_rows), labels


def make_best_hgb(random_state: int = 42) -> HistGradientBoostingClassifier:
    """Return the best fixed configuration from EXP-001."""
    return HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=7,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=random_state,
    )


def make_exp002_visible_hgb(random_state: int = 42) -> HistGradientBoostingClassifier:
    """Return EXP-002's validation-selected visible-only HGB configuration."""
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=random_state,
    )


def validate_submission_frame(submission: pd.DataFrame, expected_ids: pd.Series) -> None:
    """Validate the competition's required submission schema and probabilities."""
    if submission.columns.tolist() != ["ID", "p_up"]:
        raise ValueError("Submission columns must be exactly ['ID', 'p_up'].")
    if len(submission) != len(expected_ids):
        raise ValueError("Submission row count does not match the sample submission.")
    if not np.array_equal(submission["ID"].to_numpy(), expected_ids.to_numpy()):
        raise ValueError("Submission IDs do not exactly match the sample submission.")
    probabilities = submission["p_up"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Submission probabilities must be finite.")
    if not ((probabilities > 0.0) & (probabilities < 1.0)).all():
        raise ValueError("Submission probabilities must be strictly inside (0, 1).")
