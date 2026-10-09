"""Dense historical cases and volatility-aware features for EXP-003."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from volhack.baselines import make_feature_matrix


def prepare_source_cache() -> dict[str, np.ndarray]:
    """Read and validate supplied assets; write derived arrays only to outputs."""
    data_path = Path("epfl-vol-hack/dataset.csv")
    sample_path = Path("epfl-vol-hack/sample_submission.csv")
    returns = pd.read_csv(data_path)
    sample = pd.read_csv(sample_path)
    if returns.columns.tolist() != ["timestamp", "ret"]:
        raise ValueError("Unexpected dataset schema")
    if sample.columns.tolist() != ["ID", "p_up"] or not sample.ID.is_unique:
        raise ValueError("Unexpected sample schema or duplicate IDs")
    timestamps = returns.timestamp.to_numpy(dtype=np.int64)
    ret = returns.ret.to_numpy(dtype=float)
    ids = sample.ID.to_numpy(dtype=np.int64)
    if not np.all(np.diff(timestamps) == 60) or np.isinf(ret).any():
        raise ValueError("Invalid minute grid or infinite returns")
    rows = ((ids - timestamps[0]) // 60).astype(int)
    if (rows < 29).any() or (rows >= len(ret)).any():
        raise ValueError("Submission targets lie outside the featureable grid")
    if not np.array_equal(timestamps[rows], ids):
        raise ValueError("Submission IDs are not exact dataset timestamps")
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [data_path, sample_path]
    }
    cache = Path("outputs/cache")
    cache.mkdir(parents=True, exist_ok=True)
    hash_path = cache / "exp003_input_hashes.json"
    if hash_path.exists() and json.loads(hash_path.read_text()) != hashes:
        raise ValueError("Supplied inputs have changed since EXP-003 began")
    hash_path.write_text(json.dumps(hashes, indent=2))
    source = dict(ret=ret, timestamps=timestamps, ids=ids, rows=rows)
    np.savez(cache / "exp003_source.npz", **source)
    return source


def historical_rows(ret: np.ndarray, spacing: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Select fully observed, non-overlapping additive-return labels."""
    candidates = np.arange(750, len(ret) - 152, spacing)
    finite_prefix = np.r_[0, np.cumsum(np.isfinite(ret))]
    valid = finite_prefix[candidates + 1] - finite_prefix[candidates - 29] == 30
    rows = candidates[valid]
    # Direct summation avoids accumulated floating-point errors in a long prefix.
    move = ret[rows[:, None] - np.arange(29, -1, -1)].sum(axis=1)
    non_ties = move != 0
    return rows[non_ties], (move[non_ties] > 0).astype(int)


def window_stats(values: np.ndarray) -> np.ndarray:
    """Compute finite-observation summaries without bridging a missing stretch."""
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    safe = np.where(finite, values, 0.0)
    denominator = np.maximum(count, 1)
    mean = safe.sum(axis=1) / denominator
    std = np.sqrt(np.maximum((safe**2).sum(axis=1) / denominator - mean**2, 0))
    result = np.column_stack(
        [
            mean,
            std,
            np.abs(safe).sum(axis=1) / denominator,
            np.max(np.abs(safe), axis=1),
            (safe > 0).sum(axis=1) / denominator,
            count / values.shape[1],
        ]
    )
    result[count == 0, :5] = np.nan
    return result


def gather_window(ret: np.ndarray, rows: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Gather an observed context window, masking dataset boundaries."""
    indices = rows[:, None] + offsets
    inside = (indices >= 0) & (indices < len(ret))
    return np.where(inside, ret[np.clip(indices, 0, len(ret) - 1)], np.nan)


def feature_sets(ret: np.ndarray, rows: np.ndarray, timestamps: np.ndarray) -> dict:
    """Construct visible features and legal context that skips target-14..+91.

    Availability for a case is the complete supplied dataset. External return
    windows are allowed by the competition contract; none includes the hidden
    target interval. Normalization is computed independently within each case.
    """
    visible = ret[rows[:, None] - np.arange(29, 14, -1)]
    if not np.isfinite(visible).all():
        raise ValueError("Incomplete visible feature window")
    basic = make_feature_matrix(ret, rows)
    rms = np.maximum(np.sqrt((visible**2).mean(axis=1)), 1e-10)
    scaled = visible / rms[:, None]
    normalized = np.column_stack(
        [
            scaled.sum(axis=1) / np.sqrt(15),
            scaled[:, -3:].sum(axis=1) / np.sqrt(3),
            scaled[:, -5:].sum(axis=1) / np.sqrt(5),
            scaled[:, -10:].sum(axis=1) / np.sqrt(10),
            scaled.min(axis=1),
            scaled.max(axis=1),
            (visible > 0).mean(axis=1),
            scaled.std(axis=1),
            np.mean(np.abs(scaled), axis=1),
            np.log(rms),
            np.median(scaled, axis=1),
            np.mean(scaled**3, axis=1),
            np.mean(scaled**4, axis=1),
        ]
    )
    pre60 = window_stats(gather_window(ret, rows, -np.arange(30, 90)))
    pre240 = window_stats(gather_window(ret, rows, -np.arange(90, 330)))
    pre420 = window_stats(gather_window(ret, rows, -np.arange(330, 750)))
    post60 = window_stats(gather_window(ret, rows, np.arange(92, 152)))
    context = np.column_stack([pre60, pre240, pre420, post60])
    ratios = np.column_stack(
        [rms / np.maximum(x[:, 1], 1e-10) for x in [pre60, pre240, pre420, post60]]
    )
    minute = (timestamps[rows] // 60) % 1440
    calendar = np.column_stack(
        [np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440)]
    )
    return {
        "basic": basic,
        "normalized": normalized,
        "combined": np.column_stack([basic, normalized, scaled]),
        "pre_context": np.column_stack([basic, normalized, context[:, :18], ratios[:, :3]]),
        "full_context": np.column_stack([basic, normalized, context, ratios, calendar]),
        "gaussian": ndtr(normalized[:, 0]),
    }


def enriched_features(ret: np.ndarray, rows: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Add robust local state estimates and context-scaled observed momentum."""
    base = feature_sets(ret, rows, timestamps)["full_context"]
    visible = gather_window(ret, rows, -np.arange(29, 14, -1))
    visible_move = visible.sum(axis=1)
    extra = []
    for offsets in [
        -np.arange(30, 45),
        -np.arange(30, 150),
        np.arange(92, 107),
        np.arange(107, 212),
    ]:
        values = gather_window(ret, rows, offsets)
        stats = window_stats(values)
        empty = ~np.isfinite(values).any(axis=1)
        # Empty windows remain missing features; temporarily fill their rows
        # solely to avoid noisy all-NaN reduction warnings.
        values[empty] = 0.0
        absolute = np.abs(values)
        with np.errstate(invalid="ignore", divide="ignore"):
            quantiles = np.nanquantile(absolute, [0.50, 0.75, 0.90], axis=1).T
            rms = np.sqrt(np.nanmean(values**2, axis=1))
            scaled = values / np.maximum(rms[:, None], 1e-10)
            moments = np.column_stack([np.nanmean(scaled**3, axis=1)])
        scale = np.maximum(stats[:, 1], 1e-10)
        signal = visible_move / (scale * np.sqrt(15))
        quantiles[empty] = np.nan
        moments[empty] = np.nan
        extra.extend([stats, quantiles, moments, signal[:, None]])
    return np.column_stack([base, *extra])
