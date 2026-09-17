"""Calibrate the ground-truth replica against the platform's published stats.

This exists because the replica was wrong and the tuning built on it was
therefore aimed at the wrong target. `champion-code/metadata*.json` reports,
for every subset the platform scored, the ground truth's cluster count, noise
count and cluster size min/median/max. Those eight subsets are the only direct
observation of the real pipeline's output available, so they are the
calibration target:

    social   34-49 clusters, 13-22% noise, median 60-75, max 252-907
    arXiv    39-41 clusters, 26-29% noise, median 53-75, max 470-556

The replica was producing 41-54 clusters at 26-39% noise on social and 21-29
at 40-42% on arXiv -- far too eager to call a point noise, and on arXiv far
too coarse. Tuning against that rewarded lumping a third of the batch into one
bucket and cutting to ~20 clusters, neither of which is what the platform
scores.

Embeddings and the UMAP projection are cached per subset, so sweeping the
HDBSCAN side costs about a second per variant.

    /tmp/gtvenv/bin/python harness/calibrate_gt.py --sweep min_samples
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

CACHE = Path("/tmp/sn1_gtcache")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Measured from champion-code/metadata.json (round 52) and metadata_v2.json
# (round 53): the mean over the subsets of each kind.
TARGET = {
    "social": {"clusters": 39.8, "noise_frac": 0.168, "median": 68.2},
    "arxiv": {"clusters": 40.0, "noise_frac": 0.276, "median": 64.0},
}


def projection(texts: list[str], name: str, n_neighbors: int, n_components: int,
               seed: int) -> np.ndarray:
    """Embed and project, caching both so HDBSCAN sweeps are cheap."""
    CACHE.mkdir(parents=True, exist_ok=True)
    emb_path = CACHE / f"{name}.emb.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
    else:
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer(EMBED_MODEL).encode(
            texts, batch_size=128, show_progress_bar=False,
            normalize_embeddings=True)
        np.save(emb_path, emb)

    key = CACHE / f"{name}.umap_{n_neighbors}_{n_components}_{seed}.npy"
    if key.exists():
        return np.load(key)
    from umap import UMAP
    reduced = UMAP(n_neighbors=n_neighbors, n_components=n_components,
                   min_dist=0.0, metric="cosine",
                   random_state=seed).fit_transform(emb)
    np.save(key, reduced)
    return reduced


def bake(reduced: np.ndarray, min_cluster_size: int = 25,
         min_samples: int | None = None, method: str = "eom",
         epsilon: float = 0.0) -> np.ndarray:
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                   metric="euclidean", cluster_selection_method=method,
                   cluster_selection_epsilon=epsilon).fit_predict(reduced)


def describe(labels: np.ndarray) -> dict:
    real = labels[labels >= 0]
    sizes = np.unique(real, return_counts=True)[1] if real.size else np.array([0])
    return {"clusters": int(sizes.size) if real.size else 0,
            "noise_frac": float((labels < 0).mean()),
            "median": float(np.median(sizes)), "max": int(sizes.max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds-dir", type=Path, default=Path("/tmp/sn1_rounds"))
    parser.add_argument("--subset", action="append", default=None)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sweep", default="min_samples")
    args = parser.parse_args()

    files = sorted(args.rounds_dir.glob("round_*_subset_*.json"))
    if args.subset:
        files = [f for f in files if any(s in f.stem for s in args.subset)]

    variants: list[tuple[str, dict]] = []
    if args.sweep == "min_samples":
        for value in (None, 15, 10, 8, 5, 3, 1):
            variants.append((f"min_samples={value}", {"min_samples": value}))
    elif args.sweep == "method":
        for method in ("eom", "leaf"):
            for value in (None, 10, 5):
                variants.append((f"{method}/ms={value}",
                                 {"method": method, "min_samples": value}))
    elif args.sweep == "epsilon":
        for eps in (0.0, 0.05, 0.1, 0.2):
            variants.append((f"epsilon={eps}", {"epsilon": eps, "min_samples": 5}))
    else:
        raise SystemExit(f"unknown sweep {args.sweep}")

    rows: dict[str, list[dict]] = {}
    for path in files:
        payload = json.loads(path.read_text())
        kind = "arxiv" if "arxiv" in path.stem else "social"
        reduced = projection(list(payload["texts"]), path.stem,
                             args.n_neighbors, args.n_components, args.seed)
        for label, kwargs in variants:
            info = describe(bake(reduced, **kwargs))
            rows.setdefault(f"{kind}|{label}", []).append(info)

    for kind in ("social", "arxiv"):
        target = TARGET[kind]
        print(f"\n{kind}  target: {target['clusters']:.0f} clusters, "
              f"{target['noise_frac']:.1%} noise, median {target['median']:.0f}")
        print(f"  {'variant':<22}{'clusters':>10}{'noise':>9}{'median':>9}"
              f"{'max':>7}{'error':>9}")
        for label, _ in variants:
            got = rows.get(f"{kind}|{label}")
            if not got:
                continue
            clusters = float(np.mean([g["clusters"] for g in got]))
            noise = float(np.mean([g["noise_frac"] for g in got]))
            median = float(np.mean([g["median"] for g in got]))
            top = float(np.mean([g["max"] for g in got]))
            # Cluster count and noise share are what the submission is tuned
            # against, so they set the error; median only breaks ties.
            error = (abs(clusters - target["clusters"]) / target["clusters"]
                     + abs(noise - target["noise_frac"]) / target["noise_frac"])
            print(f"  {label:<22}{clusters:>10.1f}{noise:>8.1%}{median:>9.1f}"
                  f"{top:>7.0f}{error:>9.3f}")


if __name__ == "__main__":
    main()
