"""Check whether two-minute training sampling improves enriched EXP-003."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_exp003_search import model_for
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from volhack.final_model import enriched_features, historical_rows

CACHE = Path("outputs/cache")
CONFIG = ("enriched", 2, 15, 500, 750, 30.0)


def main() -> None:
    threadpool_limits(4)
    source = np.load(CACHE / "exp003_source.npz")
    ret, timestamps = source["ret"], source["timestamps"]
    rows, labels = historical_rows(ret, spacing=2)
    features = np.lib.format.open_memmap(
        CACHE / "exp003_density_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), enriched_features(ret, rows[:1], timestamps).shape[1]),
    )
    for start in range(0, len(rows), 10000):
        features[start : start + 10000] = enriched_features(
            ret, rows[start : start + 10000], timestamps
        )
    features.flush()
    print("Density features ready", features.shape, flush=True)
    np.savez(CACHE / "exp003_density_cases.npz", rows=rows, labels=labels)
    boundaries = (np.array([0.40, 0.55, 0.70, 0.85]) * len(ret)).astype(int)
    phase = (rows - 750) % 30 == 0
    oof = phase & (rows >= boundaries[0]) & (rows < boundaries[3])
    output, metrics = [], []
    for fold in range(3):
        lower, upper = boundaries[fold : fold + 2]
        train = rows + 211 < lower - 750
        valid = phase & (rows >= lower) & (rows < upper)
        model = model_for(CONFIG).fit(features[train], labels[train])
        p = model.predict_proba(features[valid])[:, 1]
        output.append(p)
        loss = log_loss(labels[valid], p)
        metrics.append({"model": "dense2_enriched", "fold": fold, "n": valid.sum(), "loss": loss})
        print("dense2_enriched", fold, "train", train.sum(), "loss", loss, flush=True)
    np.savez(
        CACHE / "exp003_density_oof.npz", labels=labels[oof], dense2_enriched=np.concatenate(output)
    )
    pd.DataFrame(metrics).to_csv(CACHE / "exp003_density_metrics.csv", index=False)
    (CACHE / "exp003_density_configs.json").write_text(json.dumps({"dense2_enriched": CONFIG}))


if __name__ == "__main__":
    main()
