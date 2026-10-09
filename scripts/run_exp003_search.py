"""Select an EXP-003 candidate on expanding chronological folds only."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from volhack.final_model import feature_sets, historical_rows, prepare_source_cache

CACHE = Path("outputs/cache")
CONFIGS = {
    "sparse_baseline": ("basic", 300, 7, 100, 50, 1.0),
    "dense_baseline": ("basic", 30, 7, 150, 150, 3.0),
    "dense_normalized": ("normalized", 30, 15, 200, 150, 5.0),
    "dense_combined": ("combined", 30, 15, 200, 150, 5.0),
    "dense_deeper": ("combined", 30, 31, 250, 200, 10.0),
    "dense_pre_context": ("pre_context", 30, 15, 200, 150, 5.0),
    "dense_full_context": ("full_context", 30, 15, 200, 150, 5.0),
}


def model_for(config: tuple) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=config[3],
        learning_rate=0.05,
        max_leaf_nodes=config[2],
        min_samples_leaf=config[4],
        l2_regularization=config[5],
        early_stopping=False,
        random_state=42,
    )


def main() -> None:
    threadpool_limits(4)
    source = prepare_source_cache()
    ret, timestamps = source["ret"], source["timestamps"]
    rows, labels = historical_rows(ret)
    features = feature_sets(ret, rows, timestamps)
    # Time boundaries depend on the grid, not on missingness or label outcomes.
    boundaries = (np.array([0.40, 0.55, 0.70, 0.85, 1.0]) * len(ret)).astype(int)
    folds = [(boundaries[i], boundaries[i + 1]) for i in range(3)]
    oof_mask = (rows >= boundaries[0]) & (rows < boundaries[3])
    oof_labels = labels[oof_mask]
    predictions = {"gaussian": features["gaussian"][oof_mask]}
    results = []
    for name, config in CONFIGS.items():
        start = time.monotonic()
        output = []
        for fold, (lower, upper) in enumerate(folds):
            # 750-minute precontext and +151-minute postcontext cannot cross a
            # train/validation boundary. Use the same purge for every candidate.
            train = rows + 151 < lower - 750
            if config[1] == 300:
                train &= (rows - 750) % 300 == 0
            valid = (rows >= lower) & (rows < upper)
            model = model_for(config).fit(features[config[0]][train], labels[train])
            p = model.predict_proba(features[config[0]][valid])[:, 1]
            output.append(p)
            loss = log_loss(labels[valid], p)
            results.append({"model": name, "fold": fold, "n": valid.sum(), "loss": loss})
            print(name, fold, "train", train.sum(), "valid", valid.sum(), "loss", loss, flush=True)
        predictions[name] = np.concatenate(output)
        print(name, "seconds", round(time.monotonic() - start, 1), flush=True)
    table = pd.DataFrame(results)
    table.to_csv(CACHE / "exp003_fold_metrics.csv", index=False)
    np.savez(CACHE / "exp003_oof.npz", labels=oof_labels, **predictions)
    np.savez(CACHE / "exp003_features.npz", rows=rows, labels=labels, **features)
    (CACHE / "exp003_configs.json").write_text(json.dumps(CONFIGS, indent=2))
    print(table.groupby("model").loss.agg(["mean", "max"]).sort_values("mean"), flush=True)


if __name__ == "__main__":
    main()
