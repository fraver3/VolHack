"""Fit a monotone conditional CDF with continuously sampled thresholds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_exp004_search import make_model
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from volhack.distributions import cdf_training_data, hidden_move_and_scale

CACHE = Path("outputs/cache")
CONFIG = {
    "kind": "cdf",
    "leaves": 31,
    "iterations": 500,
    "min_leaf": 500,
    "l2": 30.0,
    "continuous_thresholds": True,
}


def thresholds_for_rows(rows: np.ndarray) -> np.ndarray:
    """Deterministic thresholds independent of hidden return labels."""
    unit = (rows[:, None] * 0.6180339887498949 + np.arange(5)[None, :] * 0.2) % 1
    return (unit - 0.5) * 6


def main() -> None:
    threadpool_limits(4)
    source = np.load(CACHE / "exp003_source.npz")
    ret = source["ret"]
    cases = np.load(CACHE / "exp003_dense_features.npz")
    rows, labels = cases["rows"], cases["labels"]
    phase = (rows - 750) % 30 == 0
    rows, labels = rows[phase], labels[phase]
    base = cases["enriched"][phase].astype(np.float32)
    shape = np.load(CACHE / "exp004_shape.npy", mmap_mode="r")[phase]
    state = np.column_stack([base, shape])
    hidden, threshold, _ = hidden_move_and_scale(ret, rows)
    bounds = (np.array([0.40, 0.55, 0.70, 0.85]) * len(ret)).astype(int)
    output, metrics = [], []
    for fold in range(3):
        lower, upper = bounds[fold : fold + 2]
        train = rows + 211 < lower - 750
        valid = (rows >= lower) & (rows < upper)
        x, y = cdf_training_data(
            state[train], hidden[train], threshold[train], thresholds_for_rows(rows[train])
        )
        model = make_model(CONFIG, x.shape[1]).fit(x, y)
        p = model.predict_proba(np.column_stack([threshold[valid], state[valid]]))[:, 1]
        loss = log_loss(labels[valid], p)
        print("cdf_continuous", fold, loss, flush=True)
        metrics.append(
            {"model": "cdf_continuous", "fold": fold, "n": int(valid.sum()), "loss": loss}
        )
        output.append(p)
    mask = (rows >= bounds[0]) & (rows < bounds[3])
    np.savez(
        CACHE / "exp004_continuous_oof.npz",
        labels=labels[mask],
        cdf_continuous=np.concatenate(output),
    )
    pd.DataFrame(metrics).to_csv(CACHE / "exp004_continuous_metrics.csv", index=False)
    (CACHE / "exp004_continuous_config.json").write_text(json.dumps(CONFIG, indent=2))


if __name__ == "__main__":
    main()
