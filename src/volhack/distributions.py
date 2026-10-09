"""Conditional hidden-return distributions and asymmetric shape features."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from volhack.final_model import gather_window


def shape_features(ret: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Describe signed central mass, tails, and jumps outside the hidden span."""
    output = []
    for offsets in [
        -np.arange(29, 14, -1),
        -np.arange(30, 150),
        np.arange(92, 212),
        -np.arange(150, 750),
    ]:
        values = gather_window(ret, rows, offsets)
        finite = np.isfinite(values)
        empty = ~finite.any(axis=1)
        values[empty] = 0
        q = np.nanquantile(values, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], axis=1).T
        # Signed location is retained; do not center away the positive core.
        scale = np.maximum((q[:, 4] - q[:, 2]) / 1.349, 1e-10)
        scaled = values / scale[:, None]
        denominator = np.maximum(finite.sum(axis=1), 1)
        tails = np.column_stack(
            [
                ((scaled < threshold) & finite).sum(axis=1) / denominator
                for threshold in [-5, -3, -1, 0, 1, 3, 5]
            ]
        )
        safe = np.where(finite, values, 0)
        second = np.maximum((safe**2).sum(axis=1), 1e-20)
        downside_energy = ((np.minimum(safe, 0)) ** 2).sum(axis=1) / second
        upper = (q[:, 6] - q[:, 3]) / scale
        lower = (q[:, 3] - q[:, 0]) / scale
        central = np.where(finite, np.clip(values, q[:, 1, None], q[:, 5, None]), 0)
        trimmed_location = central.sum(axis=1) / denominator / scale
        block = np.column_stack(
            [
                q / scale[:, None],
                np.log(scale),
                tails,
                downside_energy,
                upper,
                lower,
                trimmed_location,
            ]
        )
        block[empty] = np.nan
        output.append(block)
    return np.column_stack(output)


def hidden_move_and_scale(ret: np.ndarray, rows: np.ndarray) -> tuple:
    """Return normalized hidden move and observable threshold scale."""
    visible = gather_window(ret, rows, -np.arange(29, 14, -1))
    scale = np.maximum(np.sqrt(np.mean(visible**2, axis=1) * 15), 1e-10)
    hidden = gather_window(ret, rows, -np.arange(14, -1, -1)).sum(axis=1)
    threshold = -visible.sum(axis=1) / scale
    return hidden / scale, threshold, scale


def cdf_training_data(
    state: np.ndarray, hidden: np.ndarray, actual_threshold: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Augment outcomes with explicit thresholds; learn a survival function.

    Features represent only available observations. Hidden returns are labels,
    never state features. Higher thresholds must reduce the predicted survival
    probability; the classifier imposes that monotonic constraint.
    """
    extra_thresholds = np.tile(grid, (len(state), 1)) if grid.ndim == 1 else grid
    thresholds = np.column_stack([actual_threshold, extra_thresholds])
    repeated = np.repeat(state, thresholds.shape[1], axis=0)
    x = np.column_stack([thresholds.ravel(), repeated]).astype(np.float32)
    y = (hidden[:, None] > thresholds).astype(int).ravel()
    return x, y


def empirical_convolution(ret: np.ndarray, rows: np.ndarray, power: int = 9) -> np.ndarray:
    """Bootstrap the sum of 15 draws from legal local empirical distributions.

    This preserves skew and jump mass without a Gaussian assumption. Conditional
    independence and a constant local distribution remain approximations, which
    are assessed on chronological labels before retaining this estimator.
    """
    draws = qmc.Sobol(d=15, scramble=True, seed=42).random_base2(power)
    output = []
    for start in range(0, len(rows), 256):
        block = rows[start : start + 256]
        visible = gather_window(ret, block, -np.arange(29, 14, -1))
        pool = np.column_stack(
            [
                gather_window(ret, block, -np.arange(30, 150)),
                gather_window(ret, block, np.arange(92, 212)),
                visible,
            ]
        )
        count = np.isfinite(pool).sum(axis=1)
        pool = np.sort(pool, axis=1)
        indices = (draws[None, :, :] * count[:, None, None]).astype(int)
        simulated = pool[np.arange(len(block))[:, None, None], indices].sum(axis=2)
        output.append((simulated + visible.sum(axis=1)[:, None] > 0).mean(axis=1))
    return np.clip(np.concatenate(output), 1e-6, 1 - 1e-6)
