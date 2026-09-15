"""Build evaluation rounds with ground truth from a local replica of the platform pipeline.

The competition bakes ground truth with `sentence-transformer -> UMAP -> HDBSCAN`.
Reproducing that offline is what makes submission changes measurable without
burning the 4-submissions-per-day rate limit.

`min_cluster_size=25` is not a guess: every subset in
`champion-code/metadata.json` reports a minimum ground-truth cluster size of
25-27, which pins it.

Subsets are assembled as a mixture of topical slices plus a random tail,
because Gravity tasks crawl specified topics. A uniform social sample is too
semantically homogeneous -- the real pipeline collapses it into a single
5000-point blob, which looks nothing like the 34-44 clusters with 13-29% noise
the platform reports.

Run with the ground-truth venv, not the submission venv:
    /tmp/gtvenv/bin/python harness/make_rounds.py --rounds 3
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

CORPUS_DIR = Path("/tmp/sn1_corpus")
ROUNDS_DIR = Path("/tmp/sn1_rounds")

SAMPLE_SIZE = 5000
MIN_CLUSTER_SIZE = 25
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Reproduces the reported ground-truth shape: ~40 clusters, a long tail of
# sizes (median 53-75, max 252-907), and 13-29% unclustered noise.
TOPICS_PER_SUBSET = 46
TAIL_FRACTION = 0.11
MIN_SLICE = 30
MAX_SLICE = 700


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def by_topic(rows: list[dict], min_rows: int) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[row["topic"]].append(row["text"])
    return {k: v for k, v in groups.items() if len(v) >= min_rows}


def zipf_sizes(n_topics: int, total: int, rng: random.Random) -> list[int]:
    """Skewed slice sizes so the subset has a few large topics and many small ones."""
    weights = [1.0 / (i + 1.6) ** 0.85 for i in range(n_topics)]
    rng.shuffle(weights)
    scale = total / sum(weights)
    sizes = [int(max(MIN_SLICE, min(MAX_SLICE, round(w * scale)))) for w in weights]
    return sizes


def build_social_subset(reddit: dict[str, list[str]], tweets: dict[str, list[str]],
                        n: int, rng: random.Random,
                        used: set[str]) -> tuple[list[str], list[str]]:
    """Mix Reddit and X topical slices so the median length lands in the observed
    230-330 character band, then dilute with an off-topic tail."""
    reddit_share = 0.62

    n_topics = TOPICS_PER_SUBSET
    target_core = int(n * (1.0 - TAIL_FRACTION))
    sizes = zipf_sizes(n_topics, target_core, rng)

    # The champion's router branches hard at a *preprocessed* median length of
    # 230 characters, and the real rounds sit inside its 230-330 window (all
    # three social subsets in metadata.json emit 1026 = 1000 singletons + 26
    # clusters, which only that branch produces). Preprocessing strips URLs,
    # mentions and punctuation, costing roughly 15% of the raw length, so aim
    # for a raw median near 310 to land inside the window.
    reddit = {k: [t for t in v if 160 <= len(t) <= 1200] for k, v in reddit.items()}
    tweets = {k: [t for t in v if 120 <= len(t) <= 400] for k, v in tweets.items()}
    candidates = [("reddit", k) for k in reddit if len(reddit[k]) >= MIN_SLICE] + \
        [("x", k) for k in tweets if len(tweets[k]) >= MIN_SLICE]
    rng.shuffle(candidates)

    texts: list[str] = []
    topics: list[str] = []
    # Reddit-heavy keeps the length mix inside the target window.
    want_reddit = int(n_topics * reddit_share)
    taken = {"reddit": 0, "x": 0}
    quota = {"reddit": want_reddit, "x": n_topics - want_reddit}
    for source, key in candidates:
        n_slices = taken["reddit"] + taken["x"]
        if n_slices >= n_topics:
            break
        if taken[source] >= quota[source]:
            continue
        pool = [t for t in (reddit if source == "reddit" else tweets)[key]
                if t not in used]
        if len(pool) < MIN_SLICE:
            continue
        size = min(sizes[n_slices], len(pool))
        picked = rng.sample(pool, k=size)
        used.update(picked)
        texts.extend(picked)
        topics.extend([key] * size)
        taken[source] += 1

    # The tail is drawn from topics not represented above, so the ground-truth
    # pipeline has genuine low-density points to label as noise.
    tail_need = max(0, n - len(texts))
    tail_pool: list[str] = []
    for source, key in candidates:
        if key in set(topics):
            continue
        tail_pool.extend(t for t in (reddit if source == "reddit" else tweets)[key][:40]
                         if t not in used)
        if len(tail_pool) > tail_need * 3:
            break
    tail = rng.sample(tail_pool, k=min(tail_need, len(tail_pool)))
    used.update(tail)
    texts.extend(tail)
    topics.extend(["<tail>"] * len(tail))

    order = list(range(len(texts)))
    rng.shuffle(order)
    return [texts[i] for i in order], [topics[i] for i in order]


def build_arxiv_subset(arxiv: dict[str, list[str]], n: int, rng: random.Random,
                       used: set[str]) -> tuple[list[str], list[str]]:
    """arXiv rounds are a plain random sample of unused titles."""
    pool = [(k, t) for k, v in arxiv.items() for t in v if t not in used]
    picked = rng.sample(pool, k=min(n, len(pool)))
    used.update(t for _, t in picked)
    return [t for _, t in picked], [k for k, _ in picked]


def ground_truth(texts: list[str], seed: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import HDBSCAN
    from umap import UMAP

    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(texts, batch_size=128, show_progress_bar=False,
                       normalize_embeddings=True)
    reduced = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                   metric="cosine", random_state=seed).fit_transform(emb)
    labels = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean",
                     cluster_selection_method="eom").fit_predict(reduced)
    return np.asarray(labels, dtype=np.int64)


def stats(labels: np.ndarray) -> dict:
    real = labels[labels >= 0]
    if real.size:
        _, sizes = np.unique(real, return_counts=True)
    else:
        sizes = np.array([0])
    return {
        "num_samples": int(labels.size),
        "num_clusters": int(sizes.size) if real.size else 0,
        "noise_count": int((labels < 0).sum()),
        "cluster_size_min": int(sizes.min()),
        "cluster_size_median": int(np.median(sizes)),
        "cluster_size_max": int(sizes.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROUNDS_DIR)
    args = parser.parse_args()

    reddit = by_topic(load_jsonl(CORPUS_DIR / "reddit.jsonl"), 15)
    tweets = by_topic(load_jsonl(CORPUS_DIR / "tweets.jsonl"), 15)
    arxiv = by_topic(load_jsonl(CORPUS_DIR / "arxiv.jsonl"), 1)
    print(f"usable topics: reddit={len(reddit)} x={len(tweets)} arxiv={len(arxiv)}")

    args.out.mkdir(parents=True, exist_ok=True)
    # Rounds are disjoint by construction on the platform; emulate that with a
    # single `used` set spanning every subset we generate.
    used: set[str] = set()

    for r in range(args.start_round, args.start_round + args.rounds):
        for idx in (1, 2, 3, "arxiv"):
            name = f"round_{r:04d}_subset_{idx}"
            path = args.out / f"{name}.json"
            if path.exists():
                print(f"{name}: exists, skipping")
                payload = json.loads(path.read_text())
                used.update(payload["texts"])
                continue
            rng = random.Random(args.seed * 1000 + r * 10 + (0 if idx == "arxiv" else idx))
            if idx == "arxiv":
                texts, topics = build_arxiv_subset(arxiv, args.size, rng, used)
            else:
                texts, topics = build_social_subset(reddit, tweets, args.size, rng, used)
            if len(texts) < args.size * 0.9:
                print(f"{name}: only {len(texts)} texts available, skipping")
                continue

            lengths = np.array([len(t) for t in texts])
            print(f"{name}: n={len(texts)} median_len={np.median(lengths):.0f} "
                  f"slices={len(set(topics))} -> ground truth ...", flush=True)
            labels = ground_truth(texts, seed=args.seed)
            info = stats(labels)
            print(f"  {info}", flush=True)
            path.write_text(json.dumps({
                "name": name, "texts": texts, "labels": labels.tolist(),
                "sampling_topics": topics, "stats": info,
            }))


if __name__ == "__main__":
    main()
