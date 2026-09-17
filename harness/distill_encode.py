"""Encode a corpus sample with the ground truth's sentence transformer.

Step 1 of building the submission's distilled embedding. The submission cannot
download a model, so the only way to get sentence-transformer knowledge into
the sandbox is to bake a linear approximation of it into the source file. This
script produces the regression target: teacher embeddings for a large sample
of in-domain text.

The teacher is `all-mpnet-base-v2`, and that is not a guess. The validator
source names it twice -- "text_clustering bake mode (mpnet + UMAP + HDBSCAN
peaks ~12GB)" in `common/models/api/job.py` -- and the 12GB figure fits a
768-dimension base model rather than MiniLM's 384. Everything here was built
against MiniLM first, which was wrong twice over: the blob approximated the
wrong teacher, and `make_rounds.py` baked its ground truth from the wrong
embeddings, so the replica was scoring against a pipeline the platform does
not run.

Runs in the ground-truth venv (needs torch). Feature hashing and the fit itself
happen in `distill_fit.py` under the *submission* venv, so the hashing is
guaranteed to match what runs in the sandbox.
    10|
    /tmp/gtvenv/bin/python harness/distill_encode.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

CORPUS_DIR = Path("/tmp/sn1_corpus")
OUT = Path("/tmp/sn1_distill")
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


def evaluation_texts(dirs: list[Path]) -> set[str]:
    """Every text used by any evaluation round.

    The distilled embedding predicts the very quantity the ground truth is
    derived from, so training it on the texts it is then scored on inflates the
    result. The corpus and the rounds are drawn from the same dump, so the
    overlap is otherwise total: each round's 5000 titles all sit in the
    training sample.
    """
    seen: set[str] = set()
    for directory in dirs:
        for path in sorted(directory.glob("round_*_subset_*.json")):
            seen.update(json.loads(path.read_text())["texts"])
    return seen


def sample(path: Path, count: int, seed: int, exclude: set[str]) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        texts = [json.loads(line)["text"] for line in fh]
    kept = [t for t in texts if t not in exclude]
    dropped = len(texts) - len(kept)
    if dropped:
        print(f"  dropped {dropped} texts held by evaluation rounds", flush=True)
    rng = random.Random(seed)
    if len(kept) > count:
        kept = rng.sample(kept, count)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    # mpnet costs far more than MiniLM did: ~16 texts/s on 4 CPU cores against
    # MiniLM's ~500, so the sample sizes are set by wall time rather than by
    # what helps. Fidelity was already collision-bound rather than data-bound
    # (the hash has 8192 buckets for millions of n-grams), so a smaller sample
    # is the cheap side of this trade.
    parser.add_argument("--reddit", type=int, default=40000)
    parser.add_argument("--tweets", type=int, default=60000)
    parser.add_argument("--arxiv", type=int, default=60000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--model", type=str, default=EMBED_MODEL)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--rounds-dir", action="append", type=Path,
                        default=None, help="round directories to hold out")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"teacher: {args.model}", flush=True)
    model = SentenceTransformer(args.model)

    rounds = args.rounds_dir or [Path("/tmp/sn1_rounds"), Path("/tmp/sn1_holdout")]
    exclude = evaluation_texts([d for d in rounds if d.exists()])
    print(f"holding out {len(exclude)} texts used by evaluation rounds")

    jobs = [("arxiv", args.arxiv), ("tweets", args.tweets), ("reddit", args.reddit)]
    for name, count in jobs:
        dest = args.out / f"{name}.npz"
        if dest.exists():
            print(f"{name}: exists, skipping", flush=True)
            continue
        texts = sample(CORPUS_DIR / f"{name}.jsonl", count, args.seed, exclude)
        if not texts:
            print(f"{name}: no corpus file, skipping", flush=True)
            continue
        print(f"{name}: encoding {len(texts)} texts ...", flush=True)
        emb = model.encode(texts, batch_size=args.batch, show_progress_bar=False,
                           normalize_embeddings=True).astype(np.float32)
        np.savez(dest, texts=np.array(texts, dtype=object), embeddings=emb)
        print(f"{name}: wrote {emb.shape} -> {dest}", flush=True)


if __name__ == "__main__":
    main()
