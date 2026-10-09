"""Evaluate the fixed EXP-003 selection once, then fit its final submission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_exp003_search import model_for
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from volhack.baselines import make_best_hgb, make_feature_matrix, validate_submission_frame
from volhack.final_model import (
    enriched_features,
    feature_sets,
    historical_rows,
    prepare_source_cache,
)

CACHE = Path("outputs/cache")


def training_data(config: list, source: dict) -> tuple:
    feature_name, spacing = config[:2]
    if spacing == 2 and (CACHE / "exp003_density_features.npy").exists():
        cases = np.load(CACHE / "exp003_density_cases.npz")
        return (
            cases["rows"],
            cases["labels"],
            np.load(CACHE / "exp003_density_features.npy", mmap_mode="r"),
        )
    if spacing == 5 and (CACHE / "exp003_dense_features.npz").exists():
        cases = np.load(CACHE / "exp003_dense_features.npz")
        return cases["rows"], cases["labels"], cases[feature_name]
    if spacing == 30 and (CACHE / "exp003_features.npz").exists():
        cases = np.load(CACHE / "exp003_features.npz")
        return cases["rows"], cases["labels"], cases[feature_name]
    rows, labels = historical_rows(source["ret"], spacing=spacing)
    parts = []
    for start in range(0, len(rows), 10000):
        block = rows[start : start + 10000]
        if feature_name == "enriched":
            parts.append(enriched_features(source["ret"], block, source["timestamps"]))
        else:
            parts.append(feature_sets(source["ret"], block, source["timestamps"])[feature_name])
    return rows, labels, np.concatenate(parts).astype(np.float32)


def main() -> None:
    threadpool_limits(4)
    selection = json.loads(Path("docs/exp003_final_selection.json").read_text())
    source = prepare_source_cache()
    ret, timestamps = source["ret"], source["timestamps"]
    test_boundary = int(len(ret) * 0.85)
    test_rows, test_y = historical_rows(ret, spacing=30)
    test_mask = test_rows >= test_boundary
    test_rows, test_y = test_rows[test_mask], test_y[test_mask]
    test_p, submission_p = np.zeros(len(test_rows)), np.zeros(len(source["ids"]))
    counts = {}
    for name, weight in selection["components"].items():
        config = selection["configs"][name]
        rows, labels, features = training_data(config, source)
        train = rows + 211 < test_boundary - 750
        model = model_for(config).fit(features[train], labels[train])
        if config[0] == "enriched":
            test_features = enriched_features(ret, test_rows, timestamps)
            submission_features = enriched_features(ret, source["rows"], timestamps)
        else:
            test_features = feature_sets(ret, test_rows, timestamps)[config[0]]
            submission_features = feature_sets(ret, source["rows"], timestamps)[config[0]]
        test_p += weight * model.predict_proba(test_features)[:, 1]
        print(name, "late-period evaluation complete", flush=True)
        model = model_for(config).fit(features, labels)
        submission_p += weight * model.predict_proba(submission_features)[:, 1]
        counts[name] = {"evaluation_training": int(train.sum()), "final_training": len(rows)}
        print(name, "final fit complete", counts[name], flush=True)
        del features, model
    calibration = selection["calibration"]
    for p in [test_p, submission_p]:
        p[:] = expit(
            calibration["slope"] * logit(np.clip(p, 1e-6, 1 - 1e-6)) + calibration["intercept"]
        )
        np.clip(p, 1e-6, 1 - 1e-6, out=p)
    # Benchmark is fixed; its late-period score cannot influence selection.
    baseline_rows, baseline_y = historical_rows(ret, spacing=300)
    baseline_train = baseline_rows + 211 < test_boundary - 750
    baseline = make_best_hgb().fit(
        make_feature_matrix(ret, baseline_rows[baseline_train]), baseline_y[baseline_train]
    )
    baseline_p = baseline.predict_proba(make_feature_matrix(ret, test_rows))[:, 1]
    metrics = {
        "selected_model": selection["name"],
        "test_n": len(test_y),
        "test_log_loss": log_loss(test_y, test_p),
        "test_baseline_log_loss": log_loss(test_y, baseline_p),
        "test_brier": brier_score_loss(test_y, test_p),
        "test_auc": roc_auc_score(test_y, test_p),
        "training_counts": counts,
        "selection": selection,
    }
    submission = pd.DataFrame({"ID": source["ids"], "p_up": submission_p})
    validate_submission_frame(submission, pd.Series(source["ids"]))
    output = Path("outputs/submissions/exp003_final_submission.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    # Validate the exact serialized artefact, including probability rounding.
    validate_submission_frame(pd.read_csv(output), pd.Series(source["ids"]))
    metrics.update(
        {
            "submission": str(output),
            "submission_rows": len(submission),
            "probability_min": float(submission_p.min()),
            "probability_max": float(submission_p.max()),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    )
    (CACHE / "exp003_final_report.json").write_text(json.dumps(metrics, indent=2))
    np.savez(
        CACHE / "exp003_late_predictions.npz",
        rows=test_rows,
        labels=test_y,
        selected=test_p,
        baseline=baseline_p,
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
