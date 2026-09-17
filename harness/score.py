"""The competition's scoring function.

    ari_normalized = max(0.0, ari)
    combined       = (ari_normalized + nmi) / 2

Round score is the mean of the per-subset combined scores. Ground-truth noise
carries label -1 and is treated as an ordinary label, matching what the
platform's reported `noise_count` implies.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def score_subset(true_labels, pred_labels) -> dict:
    y_true = np.asarray(true_labels, dtype=np.int64)
    y_pred = np.asarray(pred_labels, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    ari_normalized = max(0.0, ari)
    return {
        "ari": ari,
        "nmi": nmi,
        "ari_normalized": ari_normalized,
        "combined": (ari_normalized + nmi) / 2,
    }


def pred_stats(pred_labels) -> dict:
    labels = np.asarray(pred_labels, dtype=np.int64)
    real = labels[labels >= 0]
    if real.size:
        _, sizes = np.unique(real, return_counts=True)
    else:
        sizes = np.array([0])
    return {
        "num_clusters": int(sizes.size) if real.size else 0,
        "noise_count": int((labels < 0).sum()),
        "size_min": int(sizes.min()),
        "size_median": int(np.median(sizes)),
        "size_max": int(sizes.max()),
    }
