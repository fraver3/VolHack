"""Select a conservative distribution-aware blend before the late-period check."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

CACHE = Path("outputs/cache")


def main() -> None:
    old2 = np.load(CACHE / "exp003_density_oof.npz")
    old5 = np.load(CACHE / "exp003_refinement_oof.npz")
    base = (old2["dense2_enriched"] + old5["dense5_enriched_deeper"]) / 2
    predictions, configs = {"exp003": base}, {}
    z = np.load(CACHE / "exp004_oof.npz")
    labels = z["labels"]
    np.testing.assert_array_equal(labels, old2["labels"])
    predictions.update({k: z[k] for k in z.files if k != "labels"})
    configs.update(json.loads((CACHE / "exp004_configs.json").read_text()))
    z = np.load(CACHE / "exp004_continuous_oof.npz")
    np.testing.assert_array_equal(labels, z["labels"])
    predictions["cdf_continuous"] = z["cdf_continuous"]
    configs["cdf_continuous"] = json.loads((CACHE / "exp004_continuous_config.json").read_text())
    source = np.load(CACHE / "exp003_source.npz")
    cases = np.load(CACHE / "exp003_features.npz")
    bounds = (np.array([0.40, 0.55, 0.70, 0.85]) * len(source["ret"])).astype(int)
    mask = (cases["rows"] >= bounds[0]) & (cases["rows"] < bounds[3])
    rows = cases["rows"][mask]
    context = cases["full_context"][mask]
    matched = (context[:, 44] > 0.98) & (context[:, 62] == 1)
    candidates = [{"exp003": 1.0}]
    for name in predictions:
        if name == "exp003":
            continue
        candidates.extend({"exp003": 1 - w, name: w} for w in [0.1, 0.25, 0.5, 1.0])
    for cdf in ["cdf_15", "cdf_continuous"]:
        candidates.append({"exp003": 0.65, "shape_31": 0.25, cdf: 0.1})
    results = []
    for weights in candidates:
        p = sum(predictions[name] * weight for name, weight in weights.items())
        losses = [
            log_loss(labels[(rows >= a) & (rows < b)], p[(rows >= a) & (rows < b)])
            for a, b in zip(bounds[:-1], bounds[1:])
        ]
        results.append(
            {
                "weights": weights,
                "pooled": log_loss(labels, p),
                "folds": losses,
                "matched": log_loss(labels[matched], p[matched]),
            }
        )
    baseline = results[0]
    eligible = [
        r
        for r in results
        if r["matched"] <= baseline["matched"]
        and all(a <= b for a, b in zip(r["folds"], baseline["folds"]))
    ]
    winner = min(eligible, key=lambda r: r["pooled"])
    selection = {
        **winner,
        "configs": {k: configs[k] for k in winner["weights"] if k != "exp003"},
        "baseline": baseline,
        "clip": 1e-6,
        "seed": 42,
        "rule": (
            "Minimize pooled loss subject to improvement in all folds "
            "and no worsening on context-matched cases."
        ),
    }
    Path("docs/exp004_final_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (CACHE / "exp004_selection_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
