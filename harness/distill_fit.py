"""Fit and pack the submission's distilled embedding.

Step 2 of building the distilled embedding. Takes the MiniLM targets from
`distill_encode.py` and solves for the linear map

    hashed n-grams (L2-normalized, log1p counts)  ->  PCA of MiniLM space

then quantizes it, compresses it, and packs it into string literals the
submission can decode offline.

    10|Why a linear map at all: the sandbox has no internet and a 50,000-*character*
source limit, so a real transformer is out of the question. A single matrix is
the most model that fits, and it is the only component that carries knowledge
from outside the 5000-text batch. Everything else the submission computes
(BM25-LSA, PPMI) can only see the batch itself, which is why paper titles --
short, no redundancy, technical vocabulary -- score so much worse without this.

Three design choices differ from the previous submission's blob:

* The target is the PCA of the embedding space with no whitening, because
    20|  truncated PCA preserves inner products, and cosine geometry is all the
  downstream clusterer uses.
* Rows are quantized jointly by k-means (vector quantization) rather than each
  weight independently. A hash bucket's row is a direction in embedding space;
  snapping the whole direction to one of K prototypes spends the character
  budget far better than snapping each coordinate to one of four levels.
* The fit is weighted toward the domain mix the competition actually serves.

Runs in the *submission* venv, so `HashingVectorizer` hashes exactly as it will
in the sandbox.
    30|
    /tmp/venv/bin/python harness/distill_fit.py --dims 64 --word 6144 --char 2048
"""

from __future__ import annotations

import argparse
import importlib.util
import lzma
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import hstack as sparse_hstack
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

DISTILL_DIR = Path("/tmp/sn1_distill")
SUBMISSION = Path(__file__).parent.parent / "champion-code" / "code_submission_v1.py"

# 15 bits per codepoint. Base64 would give 6, and the submission limit counts
# characters, so this is a 2.5x larger model for the same budget.
CJK_BASE = 19968
BITS_PER_CHAR = 15

# The cached right-hand side is built once at this width; any smaller `--dims`
# is a prefix of it, because PCA components come out ordered by variance.
MAX_DIMS = 128


def load_clean_text():
    """Reuse the submission's own cleaner so training and inference agree."""
    spec = importlib.util.spec_from_file_location("submission_for_distill", SUBMISSION)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_for_distill"] = module
    spec.loader.exec_module(module)
    return module.clean_text


def hashed_features(docs: list[str], word_buckets: int, char_buckets: int):
    """Exactly the transform `distilled_features()` applies in the sandbox."""
    word = HashingVectorizer(n_features=word_buckets, ngram_range=(1, 2),
                             analyzer="word", alternate_sign=False, norm=None,
                             dtype=np.float32)
    blocks = [word.transform(docs)]
    if char_buckets:
        char = HashingVectorizer(n_features=char_buckets, ngram_range=(3, 5),
                                 analyzer="char_wb", alternate_sign=False,
                                 norm=None, dtype=np.float32)
        blocks.append(char.transform(docs))
    block = sparse_hstack(blocks).tocsr() if len(blocks) > 1 else blocks[0]
    block.data = np.log1p(block.data).astype(np.float32)
    return normalize(block)


def load_corpus(limits: dict[str, int], seed: int):
    texts: list[str] = []
    targets: list[np.ndarray] = []
    for name, cap in limits.items():
        path = DISTILL_DIR / f"{name}.npz"
        if cap == 0:
            print(f"  {name}: excluded")
            continue
        if not path.exists():
            print(f"  {name}: missing, skipped")
            continue
        with np.load(path, allow_pickle=True) as data:
            rows = data["texts"]
            emb = data["embeddings"]
        if len(rows) > cap:
            pick = np.random.default_rng(seed).choice(len(rows), cap, replace=False)
            rows, emb = rows[pick], emb[pick]
        print(f"  {name}: {len(rows)} texts")
        texts.extend(str(t) for t in rows)
        targets.append(emb.astype(np.float32))
    if not texts:
        raise SystemExit(f"no encoded corpus in {DISTILL_DIR}")
    return texts, np.vstack(targets)


def normal_equations(X, Y: np.ndarray, chunk: int = 2048):
    """Accumulate (X'X, X'Y).

    Both are built from dense chunks: `X` has several hundred non-zeros per
    row, so a sparse-sparse product costs sum(nnz_i^2) and loses badly to BLAS
    on moderately sized dense blocks.

    This is the expensive step and it depends only on the feature layout and
    the corpus, not on the ridge penalty, the output dimension, or the
    quantizer -- so it is cached and reused across every tuning run.
    """
    n_features = X.shape[1]
    gram = np.zeros((n_features, n_features), np.float64)
    rhs = np.zeros((n_features, Y.shape[1]), np.float64)
    for start in range(0, X.shape[0], chunk):
        block = X[start:start + chunk].toarray()
        gram += (block.T @ block).astype(np.float64)
        rhs += (block.T @ Y[start:start + chunk]).astype(np.float64)
        if (start // chunk) % 20 == 0:
            print(f"    gram {start + block.shape[0]}/{X.shape[0]}", flush=True)
    return gram, rhs


def solve_ridge(gram: np.ndarray, rhs: np.ndarray, alpha: float) -> np.ndarray:
    """Solve (X'X + alpha*tr/F*I) W = X'Y.

    The penalty is scaled by the mean diagonal so `alpha` means the same thing
    across feature layouts and corpus sizes.
    """
    n_features = gram.shape[0]
    scale = float(np.trace(gram)) / max(n_features, 1)
    regularized = gram.copy()
    regularized[np.diag_indices_from(regularized)] += alpha * max(scale, 1e-9)
    factor = cho_factor(regularized, lower=True, overwrite_a=True)
    return cho_solve(factor, rhs).astype(np.float32)


# --------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------


def quantize_magnitudes(norms: np.ndarray):
    """8-bit log-magnitude per row, with code 0 reserved for an exact zero.

    Row norms span orders of magnitude -- a bucket hit by a common word carries
    far more weight than a rare one -- so a linear scale would spend every
    level on the few largest rows.
    """
    keep = norms > 0
    positive = norms[keep]
    lo = float(np.log(positive.min())) if positive.size else 0.0
    hi = float(np.log(positive.max())) if positive.size else 1.0
    levels = np.zeros(256, np.float32)
    levels[1:] = np.exp(lo + (hi - lo) * np.arange(255) / 254.0)

    codes = np.zeros(norms.size, np.uint8)
    if positive.size:
        scaled = (np.log(positive) - lo) / max(hi - lo, 1e-12) * 254.0
        codes[keep] = np.clip(np.round(scaled), 0, 254).astype(np.uint8) + 1
    return codes, levels


def prune_rows(W: np.ndarray, fraction: float) -> np.ndarray:
    """Zero the `fraction` of rows with the smallest norm.

    Hash buckets have wildly uneven importance: most correspond to n-grams
    that barely occur, and their rows contribute almost nothing to any
    document embedding. Zeroing them is close to free in accuracy and buys a
    lot of budget, because identical all-zero code and magnitude bytes are
    exactly what the compressor is good at.
    """
    if fraction <= 0:
        return W
    norms = np.linalg.norm(W, axis=1)
    cutoff = float(np.quantile(norms, fraction))
    out = W.copy()
    out[norms <= cutoff] = 0.0
    return out


def quantize_pq(W: np.ndarray, subspaces: int, n_prototypes: int = 256,
                seed: int = 0):
    """Product-quantize the rows of W: direction by PQ, magnitude at 8 bits.

    Each row maps one hash bucket into embedding space, so it is a direction
    plus a scale. Quantizing whole directions rather than individual weights is
    what makes the character budget work: a document embedding is a weighted
    sum over hundreds of buckets, so residual direction errors average down
    instead of adding up.

    Plain vector quantization cannot be used here. A single codebook large
    enough to be accurate (K=1024 over 64 dims) costs 64 KB on its own, which
    is the entire character budget. Splitting the dimensions into `subspaces`
    independently quantized blocks fixes that: the codebooks together cost
    `256 * dims` bytes no matter how many blocks there are, while the effective
    number of representable directions is 256**subspaces.
    """
    norms = np.linalg.norm(W, axis=1)
    keep = norms > 0
    directions = np.zeros_like(W)
    directions[keep] = W[keep] / norms[keep, None]

    dims = W.shape[1]
    if dims % subspaces:
        raise SystemExit(f"dims={dims} not divisible by subspaces={subspaces}")
    width = dims // subspaces

    books = np.zeros((subspaces, n_prototypes, width), np.float32)
    codes = np.zeros((W.shape[0], subspaces), np.uint8)
    for block in range(subspaces):
        chunk = directions[:, block * width:(block + 1) * width]
        # Prototypes are learned from the rows that actually carry weight;
        # pruned rows would otherwise pull a large share of the codebook onto
        # the origin, which no real row needs.
        km = MiniBatchKMeans(n_clusters=n_prototypes, random_state=seed + block,
                             n_init=5, batch_size=4096, max_iter=200)
        km.fit(chunk[keep] if keep.any() else chunk)
        codes[:, block] = km.predict(chunk).astype(np.uint8)
        books[block] = km.cluster_centers_.astype(np.float32)
    codes[~keep] = 0

    mag_codes, levels = quantize_magnitudes(norms)
    recon = np.concatenate(
        [books[b][codes[:, b]] for b in range(subspaces)], axis=1)
    approx = recon * levels[mag_codes][:, None]
    return {"kind": "pq", "books": books, "codes": codes, "mag": mag_codes,
            "levels": levels, "subspaces": subspaces,
            "prototypes": n_prototypes}, approx.astype(np.float32)


def quantize_scalar(W: np.ndarray, bits: int):
    """Per-output-dimension scalar codebook, the previous submission's format.

    Kept only as a baseline to justify the vector-quantized format.
    """
    n_levels = 1 << bits
    books = np.zeros((W.shape[1], n_levels), np.float32)
    codes = np.zeros(W.shape, np.uint8)
    for dim in range(W.shape[1]):
        column = W[:, dim]
        quantiles = np.quantile(column, np.linspace(0, 1, n_levels + 1)[1:-1])
        assigned = np.clip(np.searchsorted(quantiles, column, side="right"),
                           0, n_levels - 1)
        for level in range(n_levels):
            members = column[assigned == level]
            books[dim, level] = float(members.mean()) if members.size else 0.0
        codes[:, dim] = assigned
    approx = np.stack([books[d][codes[:, d]] for d in range(W.shape[1])], 1)
    return {"kind": f"scalar{bits}", "books": books, "codes": codes, "bits": bits}, \
        approx.astype(np.float32)


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def pack_bits(values: np.ndarray, bits: int) -> bytes:
    """Little-endian bit packing of fixed-width unsigned codes."""
    values = np.asarray(values, np.uint32).ravel()
    if bits == 8:
        return values.astype(np.uint8).tobytes()
    out = bytearray()
    acc = held = 0
    for value in values:
        acc |= int(value) << held
        held += bits
        while held >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            held -= 8
    if held:
        out.append(acc & 0xFF)
    return bytes(out)


def serialize(payload: dict, dims: int, word_buckets: int,
              char_buckets: int) -> bytes:
    """Self-describing blob, so the submission needs no matching constants."""
    if payload["kind"] != "pq":
        raise SystemExit("only the pq format is serialized; scalar is for comparison")
    header = np.array([payload["subspaces"], payload["prototypes"], dims,
                       word_buckets, char_buckets], np.uint32).tobytes()
    # Sub-vectors of unit directions, so int8 over [-1, 1] costs ~0.004 per
    # component -- far below the error the direction quantization already has.
    books_q = np.clip(np.round(payload["books"] * 127.0), -127, 127).astype(np.int8)
    return (header + books_q.tobytes()
            + payload["levels"].astype(np.float32).tobytes()
            + payload["codes"].astype(np.uint8).tobytes()
            + payload["mag"].astype(np.uint8).tobytes())


def encode_chars(blob: bytes) -> str:
    """Pack bytes into codepoints at 15 bits each."""
    acc = bits = 0
    out: list[str] = []
    for byte in blob:
        acc = acc << 8 | byte
        bits += 8
        while bits >= BITS_PER_CHAR:
            bits -= BITS_PER_CHAR
            out.append(chr(CJK_BASE + (acc >> bits & 0x7FFF)))
    if bits:
        out.append(chr(CJK_BASE + ((acc << (BITS_PER_CHAR - bits)) & 0x7FFF)))
    return "".join(out)


def cosine_fidelity(X_probe: np.ndarray, W: np.ndarray, W_hat: np.ndarray,
                    Y_probe: np.ndarray) -> dict:
    """How well predicted embeddings reproduce MiniLM's *pairwise* geometry.

    Absolute reconstruction error is the wrong target: the clusterer only ever
    looks at cosine between pairs of documents, so what matters is agreement
    between the two pairwise similarity matrices.
    """
    truth = normalize(Y_probe)
    exact = normalize(np.asarray(X_probe @ W))
    approx = normalize(np.asarray(X_probe @ W_hat))

    def pairwise_corr(A: np.ndarray, B: np.ndarray) -> float:
        upper = np.triu_indices(A.shape[0], 1)
        return float(np.corrcoef((A @ A.T)[upper], (B @ B.T)[upper])[0, 1])

    return {
        "exact_vs_truth": pairwise_corr(exact, truth),
        "quantized_vs_truth": pairwise_corr(approx, truth),
        "quantized_vs_exact": pairwise_corr(approx, exact),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, default=64)
    parser.add_argument("--word", type=int, default=6144)
    parser.add_argument("--char", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--subspaces", type=int, default=4)
    parser.add_argument("--prototypes", type=int, default=256)
    parser.add_argument("--prune", type=float, default=0.0,
                        help="fraction of lowest-norm rows to zero")
    parser.add_argument("--reddit", type=int, default=70000)
    parser.add_argument("--tweets", type=int, default=70000)
    parser.add_argument("--arxiv", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("/tmp/sn1_distill/blob.txt"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--refit", action="store_true",
                        help="ignore any cached normal equations")
    parser.add_argument("--compare-scalar", action="store_true")
    args = parser.parse_args()

    cache = args.cache or (DISTILL_DIR / f"normal_{args.word}_{args.char}_"
                           f"{args.reddit}_{args.tweets}_{args.arxiv}.npz")
    if cache.exists() and not args.refit:
        print(f"loading cached normal equations from {cache}")
        with np.load(cache) as data:
            gram, rhs_full = data["gram"], data["rhs"]
            X_probe, Y_probe = data["x_probe"], data["y_probe"]
    else:
        clean_text = load_clean_text()
        print("loading encoded corpus")
        texts, Y = load_corpus({"reddit": args.reddit, "tweets": args.tweets,
                                "arxiv": args.arxiv}, args.seed)
        print(f"  {len(texts)} texts, target {Y.shape}")

        print("projecting target to PCA basis (no whitening preserves inner products)")
        centred = Y - Y.mean(0, keepdims=True)
        # Eigendecomposition of the 384x384 covariance beats an SVD of the
        # 260k x 384 matrix and gives components already ordered by variance,
        # so a smaller `dims` is just a prefix of the same basis.
        cov = (centred.T @ centred) / max(len(centred) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        order = np.argsort(-eigvals)[:MAX_DIMS]
        basis = eigvecs[:, order].astype(np.float32)
        print(f"  top {MAX_DIMS} of {Y.shape[1]} components keep "
              f"{eigvals[order].sum() / eigvals.sum():.1%} of variance")

        print("hashing features")
        docs = [clean_text(t) for t in texts]
        X = hashed_features(docs, args.word, args.char)
        print(f"  {X.shape}, {X.nnz / X.shape[0]:.0f} nnz/row")

        print("accumulating normal equations")
        gram, rhs_full = normal_equations(X, centred @ basis)

        # A held-out slice, kept so fidelity can be scored without rehashing
        # the whole corpus on every tuning run.
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(X.shape[0], min(4000, X.shape[0]), replace=False)
        X_probe, Y_probe = X[pick].toarray(), Y[pick]
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, gram=gram, rhs=rhs_full, x_probe=X_probe, y_probe=Y_probe)
        print(f"  cached -> {cache}")

    if args.dims > rhs_full.shape[1]:
        raise SystemExit(f"--dims {args.dims} exceeds cached {rhs_full.shape[1]}")

    print(f"solving ridge (alpha={args.alpha}, dims={args.dims})")
    W = solve_ridge(gram, rhs_full[:, :args.dims], args.alpha)
    print(f"  W {W.shape}")
    if args.prune > 0:
        W = prune_rows(W, args.prune)
        print(f"  pruned {args.prune:.0%} of rows by norm")

    print(f"product-quantizing rows: {args.subspaces} subspaces x "
          f"{args.prototypes} prototypes")
    payload, W_hat = quantize_pq(W, args.subspaces, args.prototypes, args.seed)

    blob = serialize(payload, args.dims, args.word, args.char)
    compressed = lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME)
    text = encode_chars(compressed)
    print(f"  raw {len(blob):,}B  lzma {len(compressed):,}B  "
          f"packed {len(text):,} characters")

    fidelity = cosine_fidelity(X_probe, W, W_hat, Y_probe)
    for key, value in fidelity.items():
        print(f"  {key}: {value:.4f}")

    if args.compare_scalar:
        for bits in (2, 4):
            scalar, approx = quantize_scalar(W, bits)
            corr = cosine_fidelity(X_probe, W, approx,
                                   Y_probe)["quantized_vs_exact"]
            size = len(lzma.compress(pack_bits(scalar["codes"], bits),
                                     preset=9 | lzma.PRESET_EXTREME))
            print(f"  scalar{bits}: quantized_vs_exact={corr:.4f}  "
                  f"{size * 8 // BITS_PER_CHAR:,} characters")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    meta = {"dims": args.dims, "word": args.word, "char": args.char,
            "subspaces": args.subspaces, "prototypes": args.prototypes,
            "prune": args.prune, "chars": len(text), **fidelity}
    (args.out.with_suffix(".meta.json")).write_text(repr(meta))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
