"""Audit a skew-preserving local empirical convolution without label fitting."""

from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

from volhack.distributions import empirical_convolution


def main() -> None:
    cache = Path("outputs/cache")
    source = np.load(cache / "exp003_source.npz")
    cases = np.load(cache / "exp003_features.npz")
    rows, labels = cases["rows"], cases["labels"]
    bounds = (np.array([0.40, 0.55, 0.70, 0.85]) * len(source["ret"])).astype(int)
    mask = (rows >= bounds[0]) & (rows < bounds[3])
    p = empirical_convolution(source["ret"], rows[mask])
    for fold in range(3):
        f = (rows[mask] >= bounds[fold]) & (rows[mask] < bounds[fold + 1])
        print("empirical convolution", fold, log_loss(labels[mask][f], p[f]), flush=True)
    np.savez(cache / "exp004_convolution_oof.npz", labels=labels[mask], empirical=p)


if __name__ == "__main__":
    main()
