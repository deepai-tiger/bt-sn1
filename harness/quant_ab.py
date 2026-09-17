"""Fidelity per *character*, which is not the same as fidelity per bit.

The blob is product-quantized because PQ reconstructs a row better than a
scalar codebook at equal bit width. But the budget is not bits, it is
characters after lzma, and the two quantizers compress nothing alike: an 8-bit
PQ code is a near-uniform index into 256 prototypes, so it carries almost 8
bits of real entropy and lzma can do nothing with it. A 2-bit per-dimension
code, on a matrix whose rows are mostly pruned to zero, is overwhelmingly the
same level repeated, which lzma crushes.

The top-2 submission ships `W,X,J = 8192,4096,48` at 2 bits per element, i.e.
48 dims where we buy 32, so this is worth settling rather than assuming.

Reuses the cached normal equations, so any `dims <= MAX_DIMS` is a re-solve
rather than a re-fit.

    /tmp/venv/bin/python harness/quant_ab.py --cache /tmp/sn1_mpnet/normal_16k.npz
"""

from __future__ import annotations

import argparse
import lzma
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).parent))
import distill_fit as fit  # noqa: E402


def dim_levels(W: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension codebook with an exact zero level.

    Quantiles are the wrong boundaries here. After pruning, most of a column is
    exactly zero, so several quantiles collapse onto zero and the surviving
    levels are spent describing the zero spike instead of the rows that carry
    signal. Reserving level 0 for zero and fitting the rest by Lloyd's
    algorithm on the nonzero values spends every remaining level where the
    energy is.
    """
    n_levels = 1 << bits
    books = np.zeros((W.shape[1], n_levels), np.float32)
    codes = np.zeros(W.shape, np.uint8)
    for dim in range(W.shape[1]):
        column = W[:, dim]
        nonzero = column[column != 0.0]
        if nonzero.size == 0:
            continue
        # Initialise on quantiles of the nonzero mass, then refine.
        centers = np.quantile(nonzero, np.linspace(0.0, 1.0, n_levels - 1))
        for _ in range(25):
            assigned = np.abs(nonzero[:, None] - centers[None, :]).argmin(1)
            for level in range(centers.size):
                members = nonzero[assigned == level]
                if members.size:
                    centers[level] = members.mean()
        books[dim, 1:] = centers
        picked = np.abs(column[:, None] - books[dim][None, :]).argmin(1)
        picked[column == 0.0] = 0
        codes[:, dim] = picked
    approx = np.stack([books[d][codes[:, d]] for d in range(W.shape[1])], 1)
    return {"books": books, "codes": codes}, approx.astype(np.float32)


def packed_chars(blob: bytes) -> int:
    return len(fit.encode_chars(lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--dims", type=int, action="append", default=None)
    parser.add_argument("--prune", type=float, action="append", default=None)
    args = parser.parse_args()

    with np.load(args.cache) as data:
        gram, rhs_full = data["gram"], data["rhs"]
        X_probe, Y_probe = data["x_probe"], data["y_probe"]
    rows = gram.shape[0]
    print(f"cache: {rows:,} buckets, {rhs_full.shape[1]} available dims")

    print(f"\n{'quantizer':<16}{'dims':>6}{'prune':>7}{'chars':>9}"
          f"{'fidelity':>10}{'exact':>8}")
    for dims in args.dims or [32, 48, 64]:
        W_full = fit.solve_ridge(gram, rhs_full[:, :dims], args.alpha)
        for prune in args.prune or [0.5, 0.7, 0.8]:
            W = fit.prune_rows(W_full, prune, gram, "energy")

            payload, approx = fit.quantize_pq(W, max(1, dims // 4), 256, 0)
            payload["sparse"] = True
            chars = packed_chars(fit.serialize(payload, dims, rows, 0))
            scores = fit.cosine_fidelity(X_probe, W, approx, Y_probe)
            print(f"{'pq 8-bit':<16}{dims:>6}{prune:>7.0%}{chars:>9,}"
                  f"{scores['quantized_vs_truth']:>10.4f}"
                  f"{scores['exact_vs_truth']:>8.4f}", flush=True)

            for bits in (2, 4):
                payload, approx = dim_levels(W, bits)
                blob = (payload["books"].astype(np.float32).tobytes()
                        + fit.pack_bits(payload["codes"], bits))
                chars = packed_chars(blob)
                scores = fit.cosine_fidelity(X_probe, W, approx, Y_probe)
                print(f"{f'per-dim {bits}-bit':<16}{dims:>6}{prune:>7.0%}"
                      f"{chars:>9,}{scores['quantized_vs_truth']:>10.4f}"
                      f"{scores['exact_vs_truth']:>8.4f}", flush=True)


if __name__ == "__main__":
    main()
