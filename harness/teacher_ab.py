"""How much did the wrong teacher cost?

The replica was built against `all-MiniLM-L6-v2`; the validator source names
`all-mpnet-base-v2`. Sentence-embedding spaces are usually described as
"correlated", which is the reasoning that let the assumption stand, so this
measures the correlation on the axis that actually matters.

UMAP is run with `n_neighbors=15` and HDBSCAN estimates density from the same
graph, so neither consumes cosine values directly -- both consume the *set of
nearest neighbours*. That is the statistic to compare, and it is much less
forgiving than a correlation coefficient.

Needs the same texts encoded by both teachers, i.e. two directories written by
`distill_encode.py --model ...`.

    /tmp/venv/bin/python harness/teacher_ab.py \\
        --left /tmp/sn1_distill --right /tmp/sn1_mpnet --source arxiv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, default=Path("/tmp/sn1_distill"))
    parser.add_argument("--right", type=Path, default=Path("/tmp/sn1_mpnet"))
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--sample", type=int, default=2000)
    args = parser.parse_args()

    for source in args.source or ["arxiv", "tweets", "reddit"]:
        left_path = args.left / f"{source}.npz"
        right_path = args.right / f"{source}.npz"
        if not (left_path.exists() and right_path.exists()):
            print(f"{source}: missing an encoding, skipping")
            continue

        with np.load(left_path, allow_pickle=True) as data:
            left_texts, left_emb = data["texts"], data["embeddings"]
        with np.load(right_path, allow_pickle=True) as data:
            right_texts, right_emb = data["texts"], data["embeddings"]

        index = {text: position for position, text in enumerate(left_texts)}
        pairs = [(index[text], position)
                 for position, text in enumerate(right_texts)
                 if text in index]
        if len(pairs) < args.sample:
            print(f"{source}: only {len(pairs)} shared texts, skipping")
            continue

        rng = np.random.default_rng(0)
        chosen = rng.choice(len(pairs), args.sample, replace=False)
        left = unit(left_emb[[pairs[i][0] for i in chosen]])
        right = unit(right_emb[[pairs[i][1] for i in chosen]])

        left_sim, right_sim = left @ left.T, right @ right.T
        upper = np.triu_indices(len(left), 1)
        pearson = float(np.corrcoef(left_sim[upper], right_sim[upper])[0, 1])

        print(f"\n{source}: {len(pairs)} shared texts, "
              f"{left_emb.shape[1]}d vs {right_emb.shape[1]}d")
        print(f"  pairwise-cosine correlation: {pearson:.4f}")
        for k in (15, 50):
            left_knn = np.argpartition(-left_sim, k + 1, axis=1)[:, :k + 1]
            right_knn = np.argpartition(-right_sim, k + 1, axis=1)[:, :k + 1]
            overlap = np.mean([
                len(set(left_knn[i]) & set(right_knn[i])) / (k + 1)
                for i in range(len(left))])
            note = "  <- what UMAP consumes" if k == 15 else ""
            print(f"  top-{k} neighbour overlap:    {overlap:.1%}{note}")


if __name__ == "__main__":
    main()
