"""Evaluate the fixed EXP-004 selection and write its distribution-aware blend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_exp004_continuous_cdf import thresholds_for_rows
from run_exp004_search import make_model
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from volhack.baselines import validate_submission_frame
from volhack.distributions import cdf_training_data, hidden_move_and_scale, shape_features
from volhack.final_model import enriched_features, prepare_source_cache

CACHE = Path("outputs/cache")


def state_for(ret: np.ndarray, rows: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    return np.column_stack([enriched_features(ret, rows, timestamps), shape_features(ret, rows)])


def main() -> None:
    threadpool_limits(4)
    selection = json.loads(Path("docs/exp004_final_selection.json").read_text())
    source = prepare_source_cache()
    ret, timestamps = source["ret"], source["timestamps"]
    cases = np.load(CACHE / "exp003_dense_features.npz")
    rows, labels = cases["rows"], cases["labels"]
    state = np.column_stack(
        [cases["enriched"], np.load(CACHE / "exp004_shape.npy", mmap_mode="r")]
    ).astype(np.float32)
    hidden, threshold, _ = hidden_move_and_scale(ret, rows)
    old_test = np.load(CACHE / "exp003_late_predictions.npz")
    test_rows, test_y = old_test["rows"], old_test["labels"]
    old_submission = pd.read_csv("outputs/submissions/exp003_final_submission.csv")
    validate_submission_frame(old_submission, pd.Series(source["ids"]))
    base_weight = selection["weights"].get("exp003", 0)
    test_p = base_weight * old_test["selected"]
    submission_p = base_weight * old_submission.p_up.to_numpy()
    test_state = state_for(ret, test_rows, timestamps)
    submission_state = state_for(ret, source["rows"], timestamps)
    _, test_threshold, _ = hidden_move_and_scale(ret, test_rows)
    # Submission hidden outcomes are unavailable and never accessed for labels.
    visible = ret[source["rows"][:, None] - np.arange(29, 14, -1)]
    sub_threshold = -visible.sum(axis=1) / np.maximum(
        np.sqrt((visible**2).mean(axis=1) * 15), 1e-10
    )
    counts = {}
    for name, config in selection["configs"].items():
        weight = selection["weights"][name]
        for final in [False, True]:
            train = (
                np.ones(len(rows), dtype=bool) if final else rows + 211 < int(0.85 * len(ret)) - 750
            )
            if config["kind"] == "cdf":
                train &= (rows - 750) % 30 == 0
                extra = (
                    thresholds_for_rows(rows[train])
                    if config.get("continuous_thresholds")
                    else np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
                )
                x, y = cdf_training_data(state[train], hidden[train], threshold[train], extra)
                target_x = (
                    np.column_stack([sub_threshold, submission_state])
                    if final
                    else np.column_stack([test_threshold, test_state])
                )
            else:
                x, y = state[train], labels[train]
                target_x = submission_state if final else test_state
            model = make_model(config, x.shape[1]).fit(x, y)
            p = model.predict_proba(target_x)[:, 1]
            if final:
                submission_p += weight * p
            else:
                test_p += weight * p
            counts[f"{name}_{'final' if final else 'evaluation'}"] = int(train.sum())
            print(name, "final" if final else "evaluation", "complete", flush=True)
            del x, y, model
    np.clip(test_p, 1e-6, 1 - 1e-6, out=test_p)
    np.clip(submission_p, 1e-6, 1 - 1e-6, out=submission_p)
    submission = pd.DataFrame({"ID": source["ids"], "p_up": submission_p})
    validate_submission_frame(submission, pd.Series(source["ids"]))
    path = Path("outputs/submissions/exp004_distribution_submission.csv")
    submission.to_csv(path, index=False)
    validate_submission_frame(pd.read_csv(path), pd.Series(source["ids"]))
    report = {
        "selection": selection,
        "test_log_loss": log_loss(test_y, test_p),
        "exp003_test_log_loss": log_loss(test_y, old_test["selected"]),
        "test_n": len(test_y),
        "training_counts": counts,
        "submission": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "probability_min": float(submission_p.min()),
        "probability_max": float(submission_p.max()),
    }
    (CACHE / "exp004_final_report.json").write_text(json.dumps(report, indent=2))
    np.savez(CACHE / "exp004_late_predictions.npz", rows=test_rows, labels=test_y, selected=test_p)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
