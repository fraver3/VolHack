"""Generate the validation-selected EXP-002 visible-only HGB submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from volhack.baselines import (
    PROBABILITY_EPSILON,
    make_exp002_visible_features,
    make_exp002_visible_hgb,
    make_historical_cases,
    validate_submission_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("epfl-vol-hack/dataset.csv"),
        help="Read-only minute-return CSV.",
    )
    parser.add_argument(
        "--sample-submission-path",
        type=Path,
        default=Path("epfl-vol-hack/sample_submission.csv"),
        help="Read-only sample submission CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/submissions/exp002_visible_hgb_submission.csv"),
        help="Generated submission destination.",
    )
    return parser.parse_args()


def target_rows_for_ids(ids: pd.Series, timestamps: pd.Series) -> np.ndarray:
    """Map Unix target IDs to positional rows on the regular minute grid."""
    first_timestamp = int(timestamps.iloc[0])
    target_rows = ((ids.to_numpy(dtype=np.int64) - first_timestamp) // 60).astype(int)
    if not np.all((ids.to_numpy(dtype=np.int64) - first_timestamp) % 60 == 0):
        raise ValueError("Every submission ID must lie on the dataset's minute grid.")
    if target_rows.min() < 0 or target_rows.max() >= len(timestamps):
        raise ValueError("At least one submission ID lies outside the dataset range.")
    if not np.array_equal(timestamps.iloc[target_rows].to_numpy(), ids.to_numpy()):
        raise ValueError("Submission IDs do not map exactly to dataset timestamps.")
    return target_rows


def main() -> None:
    args = parse_args()
    returns = pd.read_csv(args.data_path)
    sample_submission = pd.read_csv(args.sample_submission_path)

    if sample_submission.columns.tolist() != ["ID", "p_up"]:
        raise ValueError("Expected sample-submission columns ['ID', 'p_up'].")
    if not np.all(np.diff(returns["timestamp"].to_numpy()) == 60):
        raise ValueError("Dataset timestamps must be a regular one-minute grid.")

    training_rows, _, training_labels = make_historical_cases(returns)
    returns_array = returns["ret"].to_numpy()
    training_features = make_exp002_visible_features(returns_array, training_rows)
    model = make_exp002_visible_hgb().fit(training_features, training_labels)

    submission_rows = target_rows_for_ids(sample_submission["ID"], returns["timestamp"])
    submission_features = make_exp002_visible_features(returns_array, submission_rows)
    probabilities = model.predict_proba(submission_features)[:, 1]
    probabilities = np.clip(probabilities, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    submission = pd.DataFrame({"ID": sample_submission["ID"], "p_up": probabilities})
    validate_submission_frame(submission, sample_submission["ID"])

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_path, index=False)
    print(f"Training cases: {len(training_rows):,}")
    print(f"Submission rows: {len(submission):,}")
    print(f"Probability range: [{probabilities.min():.6f}, {probabilities.max():.6f}]")
    print(f"Wrote: {args.output_path}")


if __name__ == "__main__":
    main()
