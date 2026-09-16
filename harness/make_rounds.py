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
import json
import random
from pathlib import Path

import numpy as np

DISTILL_DIR = Path("/tmp/sn1_distill")
ROUNDS_DIR = Path("/tmp/sn1_rounds")

SAMPLE_SIZE = 5000
MIN_CLUSTER_SIZE = 25          # pinned: every reported subset has min 25-27

# Mean over the eight subsets in champion-code/metadata*.json, the only direct
# observation of the real pipeline's output we have.
TARGET = {
    "social": {"clusters": 39.8, "noise_frac": 0.168},
    "arxiv": {"clusters": 40.0, "noise_frac": 0.276},
}

# Calibrated against TARGET; see --calibrate. Deliberately central rather than
# argmax: a single seed draw swings the baked shape a long way (n_topics 35 vs
# 40 at the same tail and spread gave 22 clusters at 2% noise against 41 at
# 15%), so the argmax cell is mostly luck. What matters is that the *aggregate*
# over a round set lands on TARGET, and that individual subsets scatter around
# it the way the platform's own do -- its social subsets run 13-22% noise.
SHAPE = {
    "social": {"n_topics": 42, "tail_frac": 0.16, "spread": 2.0,
               "min_samples": None},
    "arxiv": {"n_topics": 48, "tail_frac": 0.18, "spread": 4.0,
              "min_samples": None},
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


def zipf_sizes(n_topics: int, total: int, rng: random.Random) -> list[int]:
    """A few large slices and many small ones, as the reported size spread shows.

    The platform's subsets run to a median near 68 with a maximum in the
    hundreds, so slice sizes cannot be uniform.
    """
    weights = [1.0 / (i + 1.6) ** 0.85 for i in range(n_topics)]
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
    for index, size in enumerate(sizes):
        if taken.all():
            break
        seed = int(rng.choice(available.tolist()))
        if taken[seed]:
            continue
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


def calibrate(args: argparse.Namespace) -> None:
    for kind in (["social", "arxiv"] if args.kind is None else [args.kind]):
        pool_texts, pool_emb = load_pool(SOURCES[kind])
        target = TARGET[kind]
        print(f"\n{kind}: pool {len(pool_texts)}, target "
              f"{target['clusters']:.0f} clusters, {target['noise_frac']:.1%} noise")
        print(f"  {'n_topics':>9}{'tail':>7}{'spread':>8}"
              f"{'clusters':>10}{'noise':>8}{'median':>8}{'max':>7}{'error':>8}")
        for n_topics in args.n_topics:
            for tail in args.tail:
                for spread in args.spread:
                    shape = {"n_topics": n_topics, "tail_frac": tail,
                             "spread": spread}
                    rng = random.Random(args.seed)
                    taken = np.zeros(len(pool_texts), bool)
                    indices, _ = build_subset(pool_texts, pool_emb, args.size,
                                              shape, rng, taken)
                    info = describe(ground_truth(pool_emb[indices], args.seed,
                                                 args.min_samples))
                    error = (abs(info["num_clusters"] - target["clusters"])
                             / target["clusters"]
                             + abs(info["noise_frac"] - target["noise_frac"])
                             / target["noise_frac"])
                    print(f"  {n_topics:>9}{tail:>7}{spread:>8}"
                          f"{info['num_clusters']:>10}{info['noise_frac']:>7.1%}"
                          f"{info['cluster_size_median']:>8}"
                          f"{info['cluster_size_max']:>7}{error:>8.3f}", flush=True)


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
    parser.add_argument("--min-samples", type=int, default=None)
    args = parser.parse_args()

    if args.calibrate:
        args.n_topics = args.n_topics or [30, 40, 50]
        args.tail = args.tail or [0.08, 0.16]
        args.spread = args.spread or [1.5, 2.5, 4.0]
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
