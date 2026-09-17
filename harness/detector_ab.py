"""Which ranking best identifies the ground truth's noise?

Labelling the true noise set exactly is worth +0.43 (harness/noise_oracle.py),
but a bucket only pays while it is mostly right, and our hand-rolled kNN score
reaches just ~40% precision against a ~27% base rate (harness/noise_pr.py).
That is the binding constraint on the largest lever we have.

The ground truth rejects a point when HDBSCAN finds it in a low-density region,
so HDBSCAN's own density estimates are the natural candidates to beat a
hand-rolled score. This compares them on the one thing that matters: precision
at the depth where a bucket would be cut.

    /tmp/venv/bin/python harness/detector_ab.py --subset arxiv
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).parent))
from grid import load_submission  # noqa: E402

ROUNDS_DIR = Path("/tmp/sn1_rounds")
SUBMISSION = Path(__file__).parent.parent / "champion-code" / "code_submission_v1.py"
DEPTHS = (0.05, 0.10, 0.20, 0.30)


def rankings(module, Z: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Candidate noise rankings, larger meaning more noise-like."""
    from sklearn.cluster import HDBSCAN
    from sklearn.neighbors import NearestNeighbors

    out: dict[str, np.ndarray] = {"ours (knn+centroid)":
                                  module.outlier_score(Z, labels)}

    dense = np.ascontiguousarray(Z, dtype=np.float64)
    model = HDBSCAN(min_cluster_size=25, metric="euclidean",
                    cluster_selection_method="eom", store_centers=None)
    model.fit(dense)
    # Membership strength; weak members are the ones it nearly rejected.
    # (sklearn's HDBSCAN exposes this but not GLOSH outlier scores.)
    out["hdbscan 1-prob"] = 1.0 - np.asarray(model.probabilities_, np.float64)
    out["hdbscan reject"] = (np.asarray(model.labels_) < 0).astype(np.float64)

    for k in (5, 20, 50):
        dist = NearestNeighbors(n_neighbors=min(k + 1, len(Z) - 1),
                                metric="cosine").fit(Z).kneighbors(Z)[0][:, 1:]
        out[f"knn dist k={k}"] = dist[:, -1].astype(np.float64)

    # Mean distance is a smoother density estimate than the k-th distance.
    dist = NearestNeighbors(n_neighbors=min(31, len(Z) - 1),
                            metric="cosine").fit(Z).kneighbors(Z)[0][:, 1:]
    out["knn mean k=30"] = dist.mean(1).astype(np.float64)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds-dir", type=Path, default=ROUNDS_DIR)
    parser.add_argument("--submission", type=Path, default=SUBMISSION)
    parser.add_argument("--subset", action="append", default=None)
    args = parser.parse_args()

    module = load_submission(args.submission)
    files = sorted(args.rounds_dir.glob("round_*_subset_*.json"))
    if args.subset:
        files = [f for f in files if any(s in f.stem for s in args.subset)]

    captured: dict = {}
    original = module.assign_noise

    def spy(labels, Z, sids, is_titles=False):
        captured["labels"] = np.asarray(labels).copy()
        captured["Z"] = np.asarray(Z).copy()
        return original(labels, Z, sids, is_titles)

    module.assign_noise = spy
    base = copy.deepcopy(module.CONFIG)
    module.CONFIG.update({"noise_mode": "none", "title_noise_mode": "none",
                          "extra_noise_frac": 0.0,
                          "title_extra_noise_frac": 0.0})

    table: dict[str, list[list[float]]] = {}
    rates: list[float] = []
    try:
        for path in files:
            payload = json.loads(path.read_text())
            truth = np.asarray(payload["labels"])
            noise = truth < 0
            rates.append(float(noise.mean()))
            module.cluster_texts(list(payload["texts"]))

            for name, score in rankings(module, captured["Z"],
                                        captured["labels"]).items():
                order = np.argsort(-score, kind="stable")
                row = [float(noise[order[:int(len(order) * d)]].mean())
                       for d in DEPTHS]
                table.setdefault(name, []).append(row)
            print(f"{path.stem}: done", flush=True)
    finally:
        module.CONFIG.clear()
        module.CONFIG.update(base)
        module.assign_noise = original

    print(f"\nbase rate (true noise share): {np.mean(rates):.1%}")
    print(f"\n{'ranking':<24}" + "".join(f"{f'P@{d:.0%}':>9}" for d in DEPTHS))
    for name, rows in sorted(table.items(),
                             key=lambda kv: -np.asarray(kv[1]).mean(0)[0]):
        mean = np.asarray(rows).mean(0)
        print(f"{name:<24}" + "".join(f"{v:>9.1%}" for v in mean))


if __name__ == "__main__":
    main()
