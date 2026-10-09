"""Compare asymmetric shape classifiers and conditional-CDF estimators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from volhack.distributions import cdf_training_data, hidden_move_and_scale, shape_features

CACHE = Path("outputs/cache")
CONFIGS = {
    "shape_15": {"kind": "shape", "leaves": 15, "iterations": 350, "min_leaf": 500, "l2": 20.0},
    "shape_31": {"kind": "shape", "leaves": 31, "iterations": 400, "min_leaf": 500, "l2": 20.0},
    "cdf_15": {"kind": "cdf", "leaves": 15, "iterations": 350, "min_leaf": 300, "l2": 20.0},
    "cdf_31": {"kind": "cdf", "leaves": 31, "iterations": 450, "min_leaf": 500, "l2": 30.0},
}


def make_model(config: dict, dimensions: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=config["iterations"],
        max_leaf_nodes=config["leaves"],
        min_samples_leaf=config["min_leaf"],
        l2_regularization=config["l2"],
        learning_rate=0.05,
        early_stopping=False,
        random_state=42,
        monotonic_cst=[-1] + [0] * (dimensions - 1) if config["kind"] == "cdf" else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    args = parser.parse_args()
    threadpool_limits(4)
    source = np.load(CACHE / "exp003_source.npz")
    ret = source["ret"]
    cases = np.load(CACHE / "exp003_dense_features.npz")
    rows, labels = cases["rows"], cases["labels"]
    base = cases["enriched"].astype(np.float32)
    shape = np.lib.format.open_memmap(
        CACHE / "exp004_shape.npy", mode="w+", dtype=np.float32, shape=(len(rows), 76)
    )
    for start in range(0, len(rows), 10000):
        shape[start : start + 10000] = shape_features(ret, rows[start : start + 10000])
    shape.flush()
    state = np.column_stack([base, shape]).astype(np.float32)
    hidden, threshold, scale = hidden_move_and_scale(ret, rows)
    np.savez(
        CACHE / "exp004_targets.npz",
        rows=rows,
        labels=labels,
        hidden=hidden,
        threshold=threshold,
        scale=scale,
    )
    boundaries = (np.array([0.40, 0.55, 0.70, 0.85]) * len(ret)).astype(int)
    phase = (rows - 750) % 30 == 0
    oof = phase & (rows >= boundaries[0]) & (rows < boundaries[3])
    predictions = {"labels": labels[oof]}
    metrics = []
    for name in args.models:
        config = CONFIGS[name]
        output = []
        for fold in range(3):
            lower, upper = boundaries[fold : fold + 2]
            train = rows + 211 < lower - 750
            valid = phase & (rows >= lower) & (rows < upper)
            if config["kind"] == "cdf":
                # Non-overlapping training cases with six thresholds each.
                train &= phase
                x, y = cdf_training_data(
                    state[train],
                    hidden[train],
                    threshold[train],
                    np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
                )
                valid_x = np.column_stack([threshold[valid], state[valid]])
            else:
                x, y, valid_x = state[train], labels[train], state[valid]
            model = make_model(config, x.shape[1]).fit(x, y)
            p = model.predict_proba(valid_x)[:, 1]
            output.append(p)
            loss = log_loss(labels[valid], p)
            metrics.append({"model": name, "fold": fold, "n": int(valid.sum()), "loss": loss})
            print(name, fold, "training rows", len(y), "loss", loss, flush=True)
            del x, y, model, valid_x
        predictions[name] = np.concatenate(output)
        np.savez(CACHE / "exp004_oof.npz", **predictions)
        pd.DataFrame(metrics).to_csv(CACHE / "exp004_fold_metrics.csv", index=False)
    (CACHE / "exp004_configs.json").write_text(
        json.dumps({name: CONFIGS[name] for name in args.models}, indent=2)
    )
    print(pd.DataFrame(metrics).groupby("model").loss.mean().sort_values(), flush=True)


if __name__ == "__main__":
    main()
