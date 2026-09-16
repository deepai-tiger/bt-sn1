"""Build evaluation rounds with ground truth from a replica of the platform pipeline.

The competition bakes ground truth with `sentence-transformer -> UMAP ->
HDBSCAN`. Reproducing that offline is what makes submission changes measurable
without burning the four-submissions-per-day limit.

Getting the *pipeline* right is not enough, and that is the lesson that cost a
submission. An earlier version of this file sampled subsets as a mixture of
broad topical slices, which baked to 41-54 clusters at 26-39% noise on social
and 21-29 clusters at 40-42% on arXiv. The platform reports 34-49 clusters at
13-22% and 39-41 at 26-29%. Tuning against rounds a third of whose points were
one big noise label rewarded exactly the wrong behaviour -- lump aggressively,
cut coarsely -- and the submission that won locally by 0.10 lost on the
platform by 0.10.

Two things had to change:

* **Slices are focused, not broad.** A subset is built from `n_topics` slices,
  each the nearest neighbours of a random seed post in embedding space. That
  is what crawling a topic actually returns, and it is the only way to get
  clusters the real pipeline keeps 83% of. A subreddit is not a topic; it is
  dozens of them.
* **arXiv is structured too.** It used to be a plain random sample over 154
  categories, which is diffuse enough that HDBSCAN discards 40% of it. The
  platform's arXiv subsets are no noisier than 29%, so they are not random
  samples either.

Both sources of texts reuse the MiniLM embeddings cached under
`/tmp/sn1_distill` by `distill_encode.py`, so assembling a subset and baking
its ground truth needs no re-encoding -- about 40s a subset, which is what
makes calibrating the shape knobs affordable.

    # find knobs that reproduce the platform's reported shape
    /tmp/gtvenv/bin/python harness/make_rounds.py --calibrate

    # then write rounds with them
    /tmp/gtvenv/bin/python harness/make_rounds.py --rounds 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np

DISTILL_DIR = Path("/tmp/sn1_distill")
ROUNDS_DIR = Path("/tmp/sn1_rounds")

SAMPLE_SIZE = 5000
MIN_CLUSTER_SIZE = 25          # pinned: every reported subset has min 25-27

# Mean over the twelve subsets in champion-code/metadata*.json (rounds 52-54),
# the only direct observation of the real pipeline's output we have.
#
# `size_max` is here because leaving it out cost us a submission. Matching only
# the cluster count and the noise share still allowed subsets whose largest
# cluster held 1804 of 5000 points, against a platform ceiling of 907 and a
# mean of 509. A batch with a third of its mass in one cluster rewards pulling
# every point into a cluster, so the replica recommended exactly that, and the
# recommendation did not survive contact with the real rounds.
TARGET = {
    "social": {"clusters": 40.7, "noise_frac": 0.181,
               "size_median": 66.3, "size_max": 509.0},
    "arxiv": {"clusters": 40.0, "noise_frac": 0.278,
              "size_median": 61.0, "size_max": 538.0},
}

# Calibrated against TARGET; see --calibrate. Deliberately central rather than
# argmax: a single seed draw swings the baked shape a long way (n_topics 35 vs
# 40 at the same tail and spread gave 22 clusters at 2% noise against 41 at
# 15%), so the argmax cell is mostly luck. What matters is that the *aggregate*
# over a round set lands on TARGET, and that individual subsets scatter around
# it the way the platform's own do -- its social subsets run 13-22% noise.
SHAPE = {
    # `min_sep` is the knob that fixed the replica. Without it two seeds drawn
    # into the same neighbourhood bake as one cluster, and enough collisions
    # produced subsets whose largest cluster held 1804 of 5000 points against a
    # platform maximum of 907. Requiring seeds to be mutually dissimilar brings
    # the largest cluster to ~497 against the platform's 509, and the cluster
    # count up to ~41 against its ~41, without touching the noise share.
    "social": {"n_topics": 43, "tail_frac": 0.15, "spread": 2.0,
               "min_sep": 0.35, "skew": 0.85, "min_samples": None},
    # arXiv carries noticeably more noise than social on the platform (26-29%
    # against 13-22%), and getting that share right is not cosmetic: the best
    # way to label a rejected point flips from reclaiming it to splitting it
    # off somewhere around 24%, so a replica that under-noises titles picks
    # the wrong one. An earlier setting landed at 17-21% and did exactly that.
    "arxiv": {"n_topics": 52, "tail_frac": 0.28, "spread": 6.5,
              "min_sep": 0.35, "skew": 0.85, "min_samples": None},
}


def load_pool(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Texts plus their MiniLM embeddings, concatenated over sources."""
    texts: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    for name in names:
        path = DISTILL_DIR / f"{name}.npz"
        if not path.exists():
            raise SystemExit(f"missing {path}; run harness/distill_encode.py first")
        with np.load(path, allow_pickle=True) as data:
            texts.append(data["texts"])
            embeddings.append(data["embeddings"].astype(np.float32))
    pool_texts = np.concatenate(texts)
    pool_emb = np.concatenate(embeddings)
    pool_emb /= np.linalg.norm(pool_emb, axis=1, keepdims=True) + 1e-12
    return pool_texts, pool_emb


def zipf_sizes(n_topics: int, total: int, rng: random.Random,
               skew: float = 0.85) -> list[int]:
    """A few large slices and many small ones, as the reported size spread shows.

    The platform's subsets run to a median near 66 with a maximum near 509, so
    slice sizes are neither uniform nor steeply skewed. `skew` is the knob:
    0 gives equal slices, and the tail gets heavier as it rises.
    """
    weights = [1.0 / (i + 1.6) ** skew for i in range(n_topics)]
    rng.shuffle(weights)
    scale = total / sum(weights)
    return [max(30, int(round(w * scale))) for w in weights]


def focused_slice(pool_emb: np.ndarray, seed: int, size: int, spread: float,
                  taken: np.ndarray, rng: random.Random) -> np.ndarray:
    """The neighbourhood of one seed post: what crawling a topic returns.

    `spread` widens the candidate neighbourhood before subsampling, which is
    the knob that trades cluster tightness against how much of the slice the
    ground-truth pipeline is willing to keep. At 1.0 the slice is the seed's
    `size` nearest neighbours and bakes almost noise-free; larger values mix
    in the fringe of the topic and push the noise share up toward what the
    platform reports.
    """
    width = int(min(len(pool_emb), max(size, size * spread)))
    similarity = pool_emb @ pool_emb[seed]
    similarity[taken] = -2.0
    candidates = np.argpartition(-similarity, width - 1)[:width]
    candidates = candidates[similarity[candidates] > -2.0]
    if len(candidates) <= size:
        return candidates
    return np.asarray(rng.sample(list(candidates), size))


def build_subset(pool_texts: np.ndarray, pool_emb: np.ndarray, n: int,
                 shape: dict, rng: random.Random,
                 taken: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n_topics = int(shape["n_topics"])
    core = int(n * (1.0 - float(shape["tail_frac"])))
    sizes = zipf_sizes(n_topics, core, rng)

    chosen: list[np.ndarray] = []
    topics: list[str] = []
    available = np.flatnonzero(~taken)
    # Two seeds drawn close together produce slices that the ground-truth
    # pipeline bakes as a single cluster, and enough of those collisions is how
    # a 1800-point cluster appears. Requiring seeds to be mutually dissimilar
    # keeps slices distinct, which is what holds the largest cluster near the
    # platform's ~509 while letting the count rise to its ~41.
    min_sep = float(shape.get("min_sep", 0.0))
    seeds: list[int] = []
    for index, size in enumerate(sizes):
        if taken.all():
            break
        seed = -1
        for _ in range(60):
            candidate = int(rng.choice(available.tolist()))
            if taken[candidate]:
                continue
            if min_sep > 0.0 and seeds:
                if float(np.max(pool_emb[seeds] @ pool_emb[candidate])) > min_sep:
                    continue
            seed = candidate
            break
        if seed < 0:
            continue
        seeds.append(seed)
        picked = focused_slice(pool_emb, seed, size, float(shape["spread"]),
                               taken, rng)
        if len(picked) < 30:
            continue
        taken[picked] = True
        chosen.append(picked)
        topics.extend([f"slice_{index}"] * len(picked))

    # A random tail gives the pipeline genuine low-density points to discard,
    # which is where a realistic noise share comes from.
    body = int(sum(len(c) for c in chosen))
    tail_need = max(0, n - body)
    if tail_need:
        rest = np.flatnonzero(~taken)
        tail = np.asarray(rng.sample(list(rest), min(tail_need, len(rest))))
        taken[tail] = True
        chosen.append(tail)
        topics.extend(["<tail>"] * len(tail))

    indices = np.concatenate(chosen)
    order = rng.sample(range(len(indices)), len(indices))
    return indices[order], [topics[i] for i in order]


def ground_truth(embeddings: np.ndarray, seed: int,
                 min_samples: int | None = None) -> np.ndarray:
    from sklearn.cluster import HDBSCAN
    from umap import UMAP

    reduced = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                   metric="cosine", random_state=seed).fit_transform(embeddings)
    labels = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=min_samples,
                     metric="euclidean",
                     cluster_selection_method="eom").fit_predict(reduced)
    return np.asarray(labels, dtype=np.int64)


def describe(labels: np.ndarray) -> dict:
    real = labels[labels >= 0]
    sizes = np.unique(real, return_counts=True)[1] if real.size else np.array([0])
    return {
        "num_samples": int(labels.size),
        "num_clusters": int(sizes.size) if real.size else 0,
        "noise_count": int((labels < 0).sum()),
        "noise_frac": round(float((labels < 0).mean()), 4),
        "cluster_size_min": int(sizes.min()),
        "cluster_size_median": int(np.median(sizes)),
        "cluster_size_max": int(sizes.max()),
    }


SOURCES = {"social": ["reddit", "tweets"], "arxiv": ["arxiv"]}


def shape_error(info: dict, target: dict) -> float:
    """Relative miss on all four reported statistics, weighted equally.

    Counting only clusters and noise is what let the largest-cluster error
    through, so every statistic the platform reports is scored here.
    """
    return (abs(info["num_clusters"] - target["clusters"]) / target["clusters"]
            + abs(info["noise_frac"] - target["noise_frac"]) / target["noise_frac"]
            + abs(info["cluster_size_median"] - target["size_median"])
            / target["size_median"]
            + abs(info["cluster_size_max"] - target["size_max"])
            / target["size_max"])


def calibrate(args: argparse.Namespace) -> None:
    for kind in (["social", "arxiv"] if args.kind is None else [args.kind]):
        pool_texts, pool_emb = load_pool(SOURCES[kind])
        target = TARGET[kind]
        base = SHAPE[kind]
        print(f"\n{kind}: pool {len(pool_texts)}, target {target['clusters']:.0f} "
              f"clusters, {target['noise_frac']:.1%} noise, median "
              f"{target['size_median']:.0f}, max {target['size_max']:.0f}")
        print(f"  {'topics':>7}{'skew':>6}{'min_sep':>8}{'spread':>7}{'tail':>6}"
              f"{'clusters':>10}{'noise':>8}{'median':>8}{'max':>7}{'error':>8}")
        grid = itertools.product(args.n_topics, args.skew, args.min_sep,
                                 args.spread or [base["spread"]],
                                 args.tail or [base["tail_frac"]])
        for n_topics, skew, min_sep, spread, tail in grid:
            shape = dict(base, n_topics=n_topics, skew=skew,
                         min_sep=min_sep, spread=spread,
                         tail_frac=tail)
            # One draw is too noisy to rank cells by; average a few.
            stats: list[dict] = []
            for draw in range(args.draws):
                seed = args.seed + draw
                rng = random.Random(seed)
                taken = np.zeros(len(pool_texts), bool)
                indices, _ = build_subset(pool_texts, pool_emb,
                                          args.size, shape, rng, taken)
                stats.append(describe(ground_truth(
                    pool_emb[indices], seed, args.min_samples)))
            info = {key: float(np.mean([s[key] for s in stats]))
                    for key in stats[0]}
            print(f"  {n_topics:>7}{skew:>6}{min_sep:>8}"
                  f"{shape['spread']:>7}{shape['tail_frac']:>6}"
                  f"{info['num_clusters']:>10.1f}{info['noise_frac']:>7.1%}"
                  f"{info['cluster_size_median']:>8.0f}"
                  f"{info['cluster_size_max']:>7.0f}"
                  f"{shape_error(info, target):>8.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROUNDS_DIR)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--kind", choices=("social", "arxiv"), default=None)
    parser.add_argument("--n-topics", type=int, action="append", default=None)
    parser.add_argument("--tail", type=float, action="append", default=None)
    parser.add_argument("--spread", type=float, action="append", default=None)
    parser.add_argument("--skew", type=float, action="append", default=None)
    parser.add_argument("--min-sep", type=float, action="append", default=None)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=None)
    args = parser.parse_args()

    if args.calibrate:
        args.n_topics = args.n_topics or [42, 52, 62]
        args.skew = args.skew or [0.85, 0.45]
        args.min_sep = args.min_sep or [0.0, 0.35]
        calibrate(args)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    pools = {kind: load_pool(names) for kind, names in SOURCES.items()}
    # Rounds are disjoint on the platform; one `taken` mask per pool spanning
    # every subset we generate emulates that.
    taken = {kind: np.zeros(len(texts), bool) for kind, (texts, _) in pools.items()}

    for kind, (texts, _) in pools.items():
        for path in sorted(args.out.glob("round_*_subset_*.json")):
            existing = set(json.loads(path.read_text())["texts"])
            if existing:
                taken[kind] |= np.isin(texts, list(existing))

    for round_index in range(args.start_round, args.start_round + args.rounds):
        for subset in (1, 2, 3, "arxiv"):
            name = f"round_{round_index:04d}_subset_{subset}"
            path = args.out / f"{name}.json"
            if path.exists():
                print(f"{name}: exists, skipping")
                continue
            kind = "arxiv" if subset == "arxiv" else "social"
            pool_texts, pool_emb = pools[kind]
            rng = random.Random(args.seed * 1000 + round_index * 10
                                + (0 if subset == "arxiv" else subset))
            indices, topics = build_subset(pool_texts, pool_emb, args.size,
                                           SHAPE[kind], rng, taken[kind])
            labels = ground_truth(pool_emb[indices], args.seed,
                                  SHAPE[kind]["min_samples"])
            info = describe(labels)
            print(f"{name}: {info}", flush=True)
            path.write_text(json.dumps({
                "name": name,
                "texts": [str(t) for t in pool_texts[indices]],
                "labels": labels.tolist(),
                "sampling_topics": topics,
                "stats": info,
            }))


if __name__ == "__main__":
    main()
