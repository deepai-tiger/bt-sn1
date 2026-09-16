"""Encode a corpus sample with the ground truth's sentence transformer.

Step 1 of building the submission's distilled embedding. The submission cannot
download a model, so the only way to get sentence-transformer knowledge into
the sandbox is to bake a linear approximation of it into the source file. This
script produces the regression target: MiniLM embeddings for a large sample of
in-domain text.

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
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def sample(path: Path, count: int, seed: int) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        texts = [json.loads(line)["text"] for line in fh]
    rng = random.Random(seed)
    if len(texts) > count:
        texts = rng.sample(texts, count)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reddit", type=int, default=90000)
    parser.add_argument("--tweets", type=int, default=110000)
    parser.add_argument("--arxiv", type=int, default=60000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    args.out.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(EMBED_MODEL)

    jobs = [("arxiv", args.arxiv), ("tweets", args.tweets), ("reddit", args.reddit)]
    for name, count in jobs:
        dest = args.out / f"{name}.npz"
        if dest.exists():
            print(f"{name}: exists, skipping", flush=True)
            continue
        texts = sample(CORPUS_DIR / f"{name}.jsonl", count, args.seed)
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
