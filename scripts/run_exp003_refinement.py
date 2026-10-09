"""Refine density, volatility context, and model capacity on EXP-003 folds."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_exp003_search import model_for
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from volhack.final_model import enriched_features, feature_sets, historical_rows

CACHE = Path("outputs/cache")
CONFIGS = {
    "dense5_context": ("full_context", 5, 15, 250, 300, 10.0),
    "dense5_shallow": ("full_context", 5, 7, 350, 300, 10.0),
    "dense5_deeper": ("full_context", 5, 31, 350, 500, 20.0),
    "dense5_enriched": ("enriched", 5, 15, 300, 300, 10.0),
    "dense5_enriched_deeper": ("enriched", 5, 31, 350, 500, 20.0),
    "dense5_enriched_long": ("enriched", 5, 15, 700, 500, 20.0),
}


def main() -> None:
    threadpool_limits(4)
    source = np.load(CACHE / "exp003_source.npz")
    ret, timestamps = source["ret"], source["timestamps"]
    rows, labels = historical_rows(ret, spacing=5)
    # Build in blocks to avoid large temporary matrices for long windows.
    feature_parts, enriched_parts = [], []
    for start in range(0, len(rows), 10000):
        block = rows[start : start + 10000]
        feature_parts.append(feature_sets(ret, block, timestamps)["full_context"])
        enriched_parts.append(enriched_features(ret, block, timestamps))
    features = {
        "full_context": np.concatenate(feature_parts).astype(np.float32),
        "enriched": np.concatenate(enriched_parts).astype(np.float32),
    }
    del feature_parts, enriched_parts
    boundaries = (np.array([0.40, 0.55, 0.70, 0.85]) * len(ret)).astype(int)
    # Evaluate on exactly the same non-overlapping rows as the first search.
    valid_phase = (rows - 750) % 30 == 0
    oof = valid_phase & (rows >= boundaries[0]) & (rows < boundaries[3])
    predictions = {"labels": labels[oof]}
    results = []
    for name, config in CONFIGS.items():
        start = time.monotonic()
        output = []
        for fold in range(3):
            lower, upper = boundaries[fold : fold + 2]
            train = rows + 211 < lower - 750
            valid = valid_phase & (rows >= lower) & (rows < upper)
            model = model_for(config).fit(features[config[0]][train], labels[train])
            p = model.predict_proba(features[config[0]][valid])[:, 1]
            output.append(p)
            loss = log_loss(labels[valid], p)
            results.append({"model": name, "fold": fold, "n": valid.sum(), "loss": loss})
            print(name, fold, "train", train.sum(), "loss", loss, flush=True)
        predictions[name] = np.concatenate(output)
        print(name, "seconds", round(time.monotonic() - start, 1), flush=True)
        pd.DataFrame(results).to_csv(CACHE / "exp003_refinement_metrics.csv", index=False)
        np.savez(CACHE / "exp003_refinement_oof.npz", **predictions)
    np.savez(CACHE / "exp003_dense_features.npz", rows=rows, labels=labels, **features)
    (CACHE / "exp003_refinement_configs.json").write_text(json.dumps(CONFIGS, indent=2))
    print(pd.DataFrame(results).groupby("model").loss.mean().sort_values(), flush=True)


if __name__ == "__main__":
    main()
