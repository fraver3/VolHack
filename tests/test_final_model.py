"""Availability and target-alignment checks for the final submission pipeline."""

import numpy as np
import pandas as pd
import pytest

from volhack.baselines import validate_submission_frame
from volhack.final_model import enriched_features, feature_sets, gather_window, historical_rows


def test_hidden_returns_cannot_change_features():
    ret = np.random.default_rng(42).normal(0, 0.001, 2000)
    timestamps = np.arange(2000) * 60
    rows = np.array([900])
    expected = enriched_features(ret, rows, timestamps)
    ret[886:992] = 999.0
    np.testing.assert_allclose(enriched_features(ret, rows, timestamps), expected)


def test_early_context_does_not_wrap_to_dataset_end():
    ret = np.random.default_rng(42).normal(0, 0.001, 2000)
    timestamps = np.arange(2000) * 60
    expected = feature_sets(ret, np.array([243]), timestamps)
    ret[-750:] = 999.0
    actual = feature_sets(ret, np.array([243]), timestamps)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], equal_nan=True)
    gathered = gather_window(ret, np.array([2]), np.array([-3, -2, -1, 0]))
    assert np.isnan(gathered[0, 0])
    np.testing.assert_array_equal(gathered[0, 1:], ret[:3])


def test_training_labels_require_exactly_the_full_thirty_minutes():
    ret = np.ones(2000) * 0.001
    ret[735] = np.nan  # excludes target 750 because its label starts at 721
    ret[781:811] = -0.002
    rows, labels = historical_rows(ret)
    assert 750 not in rows
    assert labels[np.where(rows == 810)[0][0]] == 0
    assert np.all(np.diff(rows) >= 30)


def test_submission_rejects_nonfinite_and_misaligned_outputs():
    ids = pd.Series([100, 200])
    valid = pd.DataFrame({"ID": ids, "p_up": [0.2, 0.8]})
    validate_submission_frame(valid, ids)
    for probabilities in [[np.nan, 0.8], [0, 0.8], [0.2, 1]]:
        with pytest.raises(ValueError):
            validate_submission_frame(valid.assign(p_up=probabilities), ids)
    with pytest.raises(ValueError):
        validate_submission_frame(valid.iloc[::-1], ids)
