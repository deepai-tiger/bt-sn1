"""Assemble and bake social subsets from the SN13 pool built by `sn13_pool.py`.

A subset draws 5000 texts from a few randomly chosen communities, with
Dirichlet-distributed shares, then bakes ground truth with the platform's
recipe (`make_rounds.ground_truth`: mpnet -> UMAP -> HDBSCAN). `--calibrate`
reports the baked shape per recipe so it can be matched to what the platform
reports; round 66's three social subsets had 36-39 clusters, 12.0-13.2%
noise, median cluster 79-90 and largest cluster 421-584.

    python3 harness/sn13_rounds.py --pool /tmp/sn13 --calibrate
    python3 harness/sn13_rounds.py --pool /tmp/sn13 --out /tmp/sn13_rounds \
        --communities 14 --alpha 2 --rounds 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_rounds import describe, ground_truth  # noqa: E402

ROUND_66 = {"clusters": 37.7, "noise_frac": 0.128, "size_median": 84.0, "size_max": 505.0}


def load(pool_dir: Path):
    pool = [json.loads(line) for line in open(pool_dir / "pool.jsonl")]
    emb = np.load(pool_dir / "pool_emb.npy")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(pool):
        # arXiv rows carry no community: the platform samples titles at random.
        groups[row.get("community", "arxiv")].append(i)
    return pool, emb, groups


def allocate(weights: np.ndarray, caps: np.ndarray, total: int) -> np.ndarray:
    counts = np.zeros(len(weights), np.int64)
    free = np.ones(len(weights), bool)
    left = total
    while left > 0 and free.any():
        share = weights * free
        share = share / share.sum()
        want = np.floor(share * left).astype(np.int64)
        want[free & (want == 0)] = 1
        room = caps - counts
        add = np.minimum(want, room)
        if add.sum() == 0:
            break
        add = np.minimum(add, left)
        counts += add
        left = total - int(counts.sum())
        free = counts < caps
    return counts


def assemble(groups, rng: random.Random, size: int, communities: int, alpha: float):
    names = rng.sample(sorted(groups), communities)
    weights = np.random.default_rng(rng.randrange(1 << 30)).dirichlet([alpha] * communities)
    caps = np.array([len(groups[c]) for c in names])
    counts = allocate(weights, caps, min(size, int(caps.sum())))
    picked = []
    for name, count in zip(names, counts, strict=True):
        picked += rng.sample(groups[name], int(count))
    rng.shuffle(picked)
    return picked, dict(zip(names, (int(c) for c in counts), strict=True))


def error(info: dict) -> float:
    t = ROUND_66
    return (abs(info["num_clusters"] - t["clusters"]) / t["clusters"]
            + abs(info["noise_frac"] - t["noise_frac"]) / t["noise_frac"]
            + abs(info["cluster_size_median"] - t["size_median"]) / t["size_median"]
            + abs(info["cluster_size_max"] - t["size_max"]) / t["size_max"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("/tmp/sn13"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/sn13_rounds"))
    parser.add_argument("--size", type=int, default=5000)
    parser.add_argument("--communities", type=int, default=14)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--subsets", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--grid-communities", type=str, default="12,16,22")
    parser.add_argument("--grid-alpha", type=str, default="1,4")
    parser.add_argument("--draws", type=int, default=2)
    args = parser.parse_args()

    pool, emb, groups = load(args.pool)
    if args.calibrate:
        grid = itertools.product([int(x) for x in args.grid_communities.split(",")],
                                 [float(x) for x in args.grid_alpha.split(",")])
        for communities, alpha in grid:
            infos = []
            for draw in range(args.draws):
                rng = random.Random(args.seed * 1000 + draw)
                picked, _ = assemble(groups, rng, args.size, communities, alpha)
                infos.append(describe(ground_truth(emb[picked], seed=draw)))
            mean = {k: float(np.mean([i[k] for i in infos])) for k in
                    ("num_clusters", "noise_frac", "cluster_size_median", "cluster_size_max")}
            print(f"communities {communities:3d} alpha {alpha:4.1f}  "
                  f"clusters {mean['num_clusters']:5.1f}  noise {mean['noise_frac']:.3f}  "
                  f"median {mean['cluster_size_median']:5.1f}  max {mean['cluster_size_max']:6.1f}  "
                  f"err {np.mean([error(i) for i in infos]):.3f}  "
                  f"per-draw {[(i['num_clusters'], i['noise_frac']) for i in infos]}", flush=True)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    for r in range(args.rounds):
        for s in range(1, args.subsets + 1):
            rng = random.Random(args.seed * 100000 + r * 100 + s)
            picked, mix = assemble(groups, rng, args.size, args.communities, args.alpha)
            labels = ground_truth(emb[picked], seed=r * 10 + s)
            info = describe(labels)
            name = f"round_{r:04d}_subset_{'arxiv' if 'arxiv' in groups else s}"
            payload = {"name": name, "texts": [pool[i]["text"] for i in picked],
                       "labels": labels.tolist(), "sampling_topics": mix, "stats": info}
            (args.out / f"{name}.json").write_text(json.dumps(payload))
            print(name, info, flush=True)


if __name__ == "__main__":
    main()
