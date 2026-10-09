"""Fix the final EXP-003 model using selection periods, before late evaluation."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

CACHE = Path("outputs/cache")


def main() -> None:
    predictions, configs = {}, {}
    labels = None
    for suffix in ["", "refinement_", "density_"]:
        path = CACHE / f"exp003_{suffix}oof.npz"
        z = np.load(path)
        if labels is None:
            labels = z["labels"]
        else:
            np.testing.assert_array_equal(labels, z["labels"])
        predictions.update({key: z[key] for key in z.files if key != "labels"})
        configs.update(json.loads((CACHE / f"exp003_{suffix}configs.json").read_text()))
    counts = pd.read_csv(CACHE / "exp003_fold_metrics.csv")
    counts = counts[counts.model == "dense_full_context"].n.to_numpy()
    edges = np.r_[0, np.cumsum(counts)]
    candidates = []
    for name, p in predictions.items():
        if name == "gaussian":
            continue
        losses = [log_loss(labels[a:b], p[a:b]) for a, b in zip(edges[:-1], edges[1:])]
        candidates.append({"name": name, "components": {name: 1.0}, "losses": losses})
    # Restrict blends to the top three individual models and fixed 50/50 weights.
    top = sorted(candidates, key=lambda item: np.mean(item["losses"]))[:3]
    for first, second in itertools.combinations(top, 2):
        a, b = first["name"], second["name"]
        p = (predictions[a] + predictions[b]) / 2
        losses = [log_loss(labels[u:v], p[u:v]) for u, v in zip(edges[:-1], edges[1:])]
        candidates.append(
            {"name": f"blend_{a}_{b}", "components": {a: 0.5, b: 0.5}, "losses": losses}
        )
    winner = min(candidates, key=lambda item: np.mean(item["losses"]))
    p = sum(predictions[name] * weight for name, weight in winner["components"].items())
    calibration_results = []
    for fold in [1, 2]:
        start, end = edges[fold : fold + 2]
        calibrator = LogisticRegression(C=1000).fit(logit(p[:start])[:, None], labels[:start])
        calibrated = calibrator.predict_proba(logit(p[start:end])[:, None])[:, 1]
        calibration_results.append(
            {
                "fold": fold,
                "raw": log_loss(labels[start:end], p[start:end]),
                "platt": log_loss(labels[start:end], calibrated),
            }
        )
    # Require forward calibration to improve both periods before retaining it.
    use_calibration = all(row["platt"] < row["raw"] for row in calibration_results)
    calibration = {"slope": 1.0, "intercept": 0.0}
    if use_calibration:
        calibrator = LogisticRegression(C=1000).fit(logit(p)[:, None], labels)
        calibration = {
            "slope": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
        }
    calibrated_p = expit(calibration["slope"] * logit(p) + calibration["intercept"])
    selection = {
        **winner,
        "configs": {name: configs[name] for name in winner["components"]},
        "calibration": calibration,
        "forward_calibration": calibration_results,
        "pooled_selection_log_loss": log_loss(labels, calibrated_p),
        "clipping_epsilon": 1e-6,
        "random_seed": 42,
        "test_boundary_fraction": 0.85,
    }
    (CACHE / "exp003_final_selection.json").write_text(json.dumps(selection, indent=2))
    Path("docs/exp003_final_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    rows = [
        dict(
            name=row["name"],
            mean=np.mean(row["losses"]),
            **{f"fold_{i}": v for i, v in enumerate(row["losses"])},
        )
        for row in candidates
    ]
    table = pd.DataFrame(rows).sort_values("mean")
    table.to_csv(CACHE / "exp003_selection_metrics.csv", index=False)
    print(table.to_string(index=False), flush=True)
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
