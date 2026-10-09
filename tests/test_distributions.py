"""Target-threshold and feature availability checks for distribution models."""

import numpy as np

from volhack.distributions import cdf_training_data, empirical_convolution, shape_features


def test_shape_features_never_observe_the_hidden_span():
    ret = np.random.default_rng(1).normal(0, 0.001, 2000)
    rows = np.array([900])
    expected = shape_features(ret, rows)
    ret[886:992] = -1000
    np.testing.assert_allclose(shape_features(ret, rows), expected)


def test_survival_labels_decrease_as_threshold_increases():
    state = np.array([[1, 2], [3, 4]])
    hidden = np.array([0.25, -0.25])
    thresholds = np.array([0.0, 0.0])
    x, y = cdf_training_data(state, hidden, thresholds, np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(x[:, 1:], np.repeat(state, 4, axis=0))
    np.testing.assert_array_equal(y.reshape(2, 4), [[1, 1, 1, 0], [0, 1, 0, 0]])


def test_convolution_respects_deterministic_return_distribution():
    ret = np.ones(2000) * 0.001
    p = empirical_convolution(ret, np.array([900]), power=5)
    assert p[0] == 1 - 1e-6
    p = empirical_convolution(-ret, np.array([900]), power=5)
    assert p[0] == 1e-6
