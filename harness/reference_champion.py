"""Readable reconstruction of the 4th-place submission, used as the A/B baseline.

`champion-code/code_submission_v1.py` as exported by the Apex CLI cannot run:
every line is padded to 200 characters, twelve lines are truncated mid-expression,
and the two `lzma`-compressed projection-matrix literals (`Ak`, `BD`) were
blanked to `'AAA'` and `''`. This file re-expresses the parts that are legible,
so the champion's *algorithm* can be measured on the same rounds as a new
submission.

Because the baked projection matrices are unrecoverable, this reconstruction
runs the champion's pipeline without them -- which is also the situation any new
submission starts from. Comparisons against it are therefore apples-to-apples
on everything except the distilled embedding.

Reconstructed from: feature builders `t`/`Aw`/`BL`/`BM`, post-processing
`s`/`Az`/`BT`/`A_`, density estimators `B0`/`BU`/`B1`, clustering via
`scipy.cluster.hierarchy.linkage(..., 'average', 'cosine')` + `fcluster`, and
the label surgery in `A5`/`At`/`Au`/`RED`/`MNN`.
"""

from __future__ import annotations

import os
import re

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.sparse import coo_matrix, csr_matrix, hstack  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

sklearn.set_config(working_memory=128)

import time  # noqa: E402

TIME_BUDGET = 70.0
SIF_A = 0.001
PPMI_WINDOW = 10
PPMI_DIM = 128
LSA_DIM = 192
ANISOTROPY_HI = 0.045
ANISOTROPY_LO = 0.03
CLUSTERS_RANGE = (40, 300)
NOISE_RANGE = (0.25, 0.40)
REL_RANGE = (0.10, 0.32)
TITLE_CLUSTERS = 80
TITLE_NOISE = 0.40
NOISE_ID_BASE = 1_000_000

_RE_URL = re.compile(r"http\S+|www\S+|https\S+")
_RE_MENTION = re.compile(r"@\w+")
_RE_HASH = re.compile(r"#(\w+)")
_RE_PUNCT = re.compile(r"[^\w\s]")
_RE_WS = re.compile(r"\s+")
_RE_RT = re.compile(r"^rt\s+", re.IGNORECASE)
_RE_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_RE_REPEAT = re.compile(r"(.)\1{2,}")
_RE_WORD = re.compile(r"[^\W\d_][\w\-']*", re.UNICODE)

STOPWORDS = set(
    "a an the of for and or to in on with by from at as is are be been we our us "
    "this that it its i you your he she they them his her their what which who "
    "when where how all any both each few more".split()
)

SCRIPT_RANGES = (
    (1024, 1327, 1), (19968, 40959, 2), (12352, 12543, 2), (44032, 55215, 2),
    (1536, 1791, 3), (1424, 1535, 4), (3584, 3711, 5), (2304, 2431, 6),
)


def preprocess(text: str) -> str:
    text = text.lower()
    text = _RE_URL.sub(" ", text)
    text = _RE_MENTION.sub(" ", text)
    text = _RE_HASH.sub(r"\1", text)
    text = _RE_PUNCT.sub(" ", text)
    return _RE_WS.sub(" ", text).strip()


def clean_rich(text: str) -> str:
    text = _RE_URL.sub(" ", str(text))
    text = _RE_RT.sub(" ", text)
    text = _RE_MENTION.sub(" ", text)
    text = _RE_HASH.sub(r"\1", text)
    text = _RE_CAMEL.sub(" ", text)
    text = _RE_REPEAT.sub(r"\1\1", text)
    return _RE_WS.sub(" ", text).strip()[:4000]


def script_id(text: str) -> int:
    counts = [0] * 7
    for ch in text:
        code = ord(ch)
        for lo, hi, sid in SCRIPT_RANGES:
            if lo <= code <= hi:
                counts[sid] += 1
                break
    best = max(range(1, 7), key=lambda i: counts[i])
    return best if counts[best] * 8 >= (len(text) or 1) else 0


def remove_top_components(X: np.ndarray, n: int = 1) -> np.ndarray:
    X = np.asarray(X, np.float32)
    if min(X.shape) < 3 or n < 1:
        return normalize(X)
    X = X - X.mean(0, keepdims=True)
    n = min(int(n), X.shape[1] - 1, X.shape[0] - 1)
    if n < 1:
        return normalize(X)
    svd = TruncatedSVD(n_components=n, random_state=42).fit(X)
    return normalize(X - svd.inverse_transform(svd.transform(X)))


def anisotropy(X: np.ndarray) -> float:
    X = np.asarray(X, np.float32)
    if min(X.shape) < 3:
        return 0.0
    X = X - X.mean(0, keepdims=True)
    return float(TruncatedSVD(n_components=1, random_state=42)
                 .fit(X).explained_variance_ratio_[0])


def lsa_features(docs: list[str]) -> np.ndarray | None:
    """TF-IDF over word 1-2 grams plus char_wb 3-5 grams, reduced by SVD."""
    for min_df in ({True: 2, False: 1}[len(docs) >= 50], 1):
        blocks = []
        for kwargs in (dict(stop_words="english", ngram_range=(1, 2), analyzer="word"),
                       dict(analyzer="char_wb", ngram_range=(3, 5))):
            try:
                mat = TfidfVectorizer(max_features=100000, min_df=min_df, max_df=0.5,
                                      sublinear_tf=True, dtype=np.float32,
                                      **kwargs).fit_transform(docs)
                if mat.shape[1] >= 2:
                    blocks.append(normalize(mat))
            except ValueError:
                continue
        if blocks:
            mat = hstack(blocks).tocsr() if len(blocks) > 1 else blocks[0]
            dim = min(LSA_DIM, mat.shape[1] - 1, max(2, mat.shape[0] - 1))
            if dim < 2:
                return None
            return normalize(TruncatedSVD(n_components=dim, random_state=42)
                             .fit_transform(mat))
    return None


def ppmi_window_features(docs: list[str], sif: bool = False) -> np.ndarray | None:
    """Word vectors from a PPMI-weighted sliding-window co-occurrence matrix,
    composed into document vectors by SIF or TF-IDF weighting."""
    try:
        counter = CountVectorizer(min_df=3, stop_words="english", dtype=np.int32,
                                  max_features=20000)
        counter.fit(docs)
    except ValueError:
        return None
    vocab = counter.vocabulary_
    if len(vocab) < 10:
        return None

    analyzer = counter.build_analyzer()
    rows, cols = [], []
    for doc in docs:
        ids = [vocab[tok] for tok in analyzer(doc) if tok in vocab]
        for pos, left in enumerate(ids):
            for right in ids[pos + 1: pos + 1 + PPMI_WINDOW]:
                rows.append(left)
                cols.append(right)
                rows.append(right)
                cols.append(left)
    if not rows:
        return None

    size = len(vocab)
    co = coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                    shape=(size, size)).tocsr()
    co.sum_duplicates()
    total = co.sum()
    if total <= 0:
        return None
    row_sum = np.asarray(co.sum(axis=1)).ravel()
    col_sum = np.asarray(co.sum(axis=0)).ravel()
    co = co.tocoo()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(co.data * total / (row_sum[co.row] * col_sum[co.col] + 1e-12) + 1e-12)
    pmi[~np.isfinite(pmi)] = 0.0
    keep = pmi > 0
    if not keep.any():
        return None

    ppmi = coo_matrix((pmi[keep], (co.row[keep], co.col[keep])),
                      shape=(size, size)).tocsr()
    dim = min(PPMI_DIM, max(2, min(ppmi.shape) - 1))
    word_vecs = normalize(TruncatedSVD(n_components=dim, random_state=42)
                          .fit_transform(ppmi))

    if sif:
        counts = counter.transform(docs).astype(np.float32)
        freq = np.asarray(counts.sum(0)).ravel()
        freq = freq / max(float(freq.sum()), 1.0)
        weights = (SIF_A / (SIF_A + freq)).astype(np.float32)
        return normalize(np.asarray(counts.multiply(weights) @ word_vecs))
    tfidf = TfidfVectorizer(vocabulary=vocab, stop_words="english",
                            sublinear_tf=True, dtype=np.float32)
    return normalize(tfidf.fit_transform(docs) @ word_vecs)


def bm25(counts, k1: float = 3.0, b: float = 0.75):
    counts = counts.tocsr().astype(np.float32)
    doc_len = np.asarray(counts.sum(1)).ravel()
    avg_len = max(float(doc_len.mean()), 1e-9)
    df = np.asarray((counts > 0).sum(0)).ravel()
    idf = np.log(1.0 + (counts.shape[0] - df + 0.5) / (df + 0.5)).astype(np.float32)
    out = counts.tocoo()
    denom = out.data + k1 * (1.0 - b + b * doc_len[out.row] / avg_len)
    data = (out.data * (k1 + 1.0) / denom) * idf[out.col]
    return coo_matrix((data.astype(np.float32), (out.row, out.col)),
                      shape=counts.shape).tocsr()


def bm25_features(docs: list[str], char_weight: float = 0.35,
                  char_features: int = 8000) -> np.ndarray | None:
    blocks = []
    try:
        blocks.append(bm25(CountVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=20000,
            stop_words="english", dtype=np.float32).fit_transform(docs)))
    except ValueError:
        pass
    try:
        char_block = bm25(CountVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.95,
            max_features=char_features, dtype=np.float32).fit_transform(docs))
        if char_block.shape[1] > 0:
            blocks.append(char_block * char_weight)
    except ValueError:
        pass
    if not blocks:
        return None
    mat = normalize(blocks[0] if len(blocks) == 1 else hstack(blocks).tocsr())
    dim = int(min(LSA_DIM, mat.shape[1] - 1, len(docs) - 1))
    if dim < 2:
        return None
    return normalize(TruncatedSVD(dim, random_state=0)
                     .fit_transform(mat)).astype(np.float32)


def ppmi_doc_features(texts: list[str]) -> np.ndarray | None:
    """Word vectors from document-level (set-of-words) PPMI co-occurrence."""
    tokens = [[t for t in (m.group(0).lower() for m in _RE_WORD.finditer(clean_rich(x)))
               if len(t) > 2 and t not in STOPWORDS] for x in texts]
    df: dict[str, int] = {}
    for toks in tokens:
        for tok in set(toks):
            df[tok] = df.get(tok, 0) + 1
    vocab_list = [w for w, c in df.items() if c >= 3]
    vocab_list.sort(key=lambda w: (-df[w], w))
    vocab_list = vocab_list[:6000]
    if len(vocab_list) < 50:
        return None

    index = {w: i for i, w in enumerate(vocab_list)}
    rows, cols = [], []
    for doc_id, toks in enumerate(tokens):
        for tok in set(toks):
            col = index.get(tok)
            if col is not None:
                rows.append(doc_id)
                cols.append(col)
    if not rows:
        return None

    doc_term = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          shape=(len(tokens), len(index)))
    co = (doc_term.T @ doc_term).toarray().astype(np.float32)
    np.fill_diagonal(co, 0.0)
    total = float(co.sum())
    if total <= 0:
        return None
    row_sum = co.sum(1, keepdims=True)
    col_sum = co.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(co * (total / (row_sum * col_sum + 1e-12)) + 1e-12, dtype=np.float32)
    del co
    pmi[~np.isfinite(pmi)] = 0.0
    np.maximum(pmi, 0.0, out=pmi)
    dim = int(min(96, len(index) - 1))
    if dim < 2:
        return None
    word_vecs = normalize(TruncatedSVD(dim, random_state=0).fit_transform(pmi))
    del pmi
    idf = np.log(len(tokens) / (1.0 + np.asarray(doc_term.sum(0)).ravel()))
    docs = normalize(doc_term.multiply(idf).tocsr() @ word_vecs).astype(np.float32)
    dead = np.abs(docs).sum(1) < 1e-9
    if dead.any():
        docs[dead] = np.random.RandomState(0).normal(
            size=(int(dead.sum()), docs.shape[1])) * 0.001
    return docs


def knn_smooth(Z: np.ndarray, k: int = 12, alpha: float = 0.2,
               iters: int = 1) -> np.ndarray:
    n = Z.shape[0]
    k_eff = min(k + 1, n - 1)
    if k_eff < 2:
        return Z
    try:
        nbr = NearestNeighbors(n_neighbors=k_eff, metric="cosine").fit(Z)
        idx = nbr.kneighbors(Z)[1][:, 1:]
    except Exception:  # noqa: BLE001
        return Z
    for _ in range(iters):
        avg = np.empty_like(Z)
        for start in range(0, n, 2000):
            avg[start:start + 2000] = Z[idx[start:start + 2000]].mean(axis=1)
        Z = normalize((1.0 - alpha) * Z + alpha * avg)
    return Z.astype(np.float32)


def spectral_features(Z: np.ndarray, dim: int = 32, k: int = 15) -> np.ndarray | None:
    n = Z.shape[0]
    if n < 50 or dim <= 0:
        return None
    try:
        graph = NearestNeighbors(n_neighbors=min(k, n - 1), metric="cosine") \
            .fit(Z).kneighbors_graph(Z, mode="connectivity")
        graph = (graph + graph.T > 0).astype(np.float32)
        deg = np.asarray(graph.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        inv = 1.0 / np.sqrt(deg)
        norm_adj = csr_matrix(graph.multiply(inv).T.multiply(inv).T)
        dim = min(dim, n - 2)
        vals, vecs = eigsh(norm_adj, k=dim + 1, which="LA")
        order = np.argsort(-vals)
        return normalize(vecs[:, order[1:]].astype(np.float32))
    except Exception:  # noqa: BLE001
        return None


def relative_density(Z: np.ndarray, k: int = 12, sample: int = 1500):
    """Mean kNN distance relative to mean pairwise distance; drives cluster count."""
    nbr = NearestNeighbors(n_neighbors=min(k + 1, len(Z) - 1), metric="cosine").fit(Z)
    knn_dist = nbr.kneighbors(Z)[0][:, 1:]
    local = float(knn_dist[:, -1].mean())
    pick = np.random.default_rng(0).choice(len(Z), min(sample, len(Z)), replace=False)
    glob = float((1.0 - Z[pick] @ Z[pick].T).mean())
    return (local / max(glob, 1e-9), knn_dist)


def plan_from_density(rel: float) -> tuple[int, float]:
    frac = float(np.clip((rel - REL_RANGE[0]) / (REL_RANGE[1] - REL_RANGE[0]), 0.0, 1.0))
    n_clusters = int(round(CLUSTERS_RANGE[0] +
                           (CLUSTERS_RANGE[1] - CLUSTERS_RANGE[0]) * frac))
    noise = float(NOISE_RANGE[0] + (NOISE_RANGE[1] - NOISE_RANGE[0]) * frac)
    return n_clusters, noise


def merge_by_script(labels: np.ndarray, sids: np.ndarray, lo: float = 0.05,
                    hi: float = 0.4, purity: float = 0.55, min_size: int = 3):
    """Collapse the per-script clusters an English-centric embedding produces."""
    out = np.asarray(labels, np.int64).copy()
    for sid in range(1, 7):
        mask = sids == sid
        share = float(mask.mean()) if out.size else 0.0
        if share < lo or share > hi:
            continue
        targets = []
        for label in np.unique(out[mask]):
            member = out == label
            if member.sum() >= min_size and (sids[member] == sid).mean() >= purity:
                targets.append(int(label))
        if len(targets) >= 2:
            out[np.isin(out, targets)] = targets[0]
    return out


def protect_cohesive(scores: np.ndarray, labels: np.ndarray, Z: np.ndarray, n: int,
                     cap: int = 3, lo: float = 0.012, hi: float = 0.12,
                     min_size: int = 40) -> np.ndarray:
    """Make the best-supported mid-size clusters ineligible for noise promotion."""
    ranked = []
    for label in np.unique(labels):
        mask = labels == label
        size = int(mask.sum())
        if size < max(min_size, int(lo * n)) or size > int(hi * n):
            continue
        centroid = Z[mask].mean(0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        ranked.append((size * float((Z[mask] @ centroid).mean()), mask))
    ranked.sort(key=lambda item: -item[0])
    for _, mask in ranked[:cap]:
        scores[mask] = -1.0
    return scores


def merge_closest_pair(docs: list[str], labels: np.ndarray,
                       thr: float = 0.7) -> np.ndarray:
    out = np.asarray(labels, np.int64).copy()
    ids = np.unique(out)
    joined = [" ".join(np.asarray(docs)[out == i][:80]) for i in ids]
    try:
        mat = normalize(TfidfVectorizer(max_features=4000, stop_words="english",
                                        dtype=np.float32).fit_transform(joined))
    except ValueError:
        return out
    sim = (mat @ mat.T).toarray()
    np.fill_diagonal(sim, 0)
    a, b = divmod(int(sim.argmax()), len(ids))
    if sim[a, b] >= thr:
        out[out == ids[b]] = ids[a]
    return out


def absorb_small(labels: np.ndarray, Z: np.ndarray, thr: float = 0.76,
                 min_size: int = 5) -> np.ndarray:
    labels = np.asarray(labels, np.int64).copy()
    ids, counts = np.unique(labels, return_counts=True)
    big, small = ids[counts >= min_size], ids[counts < min_size]
    if big.size < 1 or small.size < 1:
        return labels
    centroids = normalize(np.stack([Z[labels == i].mean(0) for i in big]))
    for label in small:
        mask = labels == label
        sim = Z[mask] @ centroids.T
        best, value = sim.argmax(1), sim.max(1)
        for pos, row in enumerate(np.where(mask)[0]):
            if value[pos] >= thr:
                labels[row] = big[best[pos]]
    return labels


def merge_mutual_nn(labels: np.ndarray, Z: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, np.int64).copy()
    for thr, max_size, rounds in ((0.8, 1, 1), (0.86, 5, 4)):
        for _ in range(rounds):
            ids, counts = np.unique(labels, return_counts=True)
            small = ids[counts <= max_size]
            if small.size < 2:
                break
            centroids = normalize(np.stack([Z[labels == i].mean(0) for i in small]))
            dist, idx = NearestNeighbors(n_neighbors=2, metric="cosine") \
                .fit(centroids).kneighbors(centroids)
            merged = False
            for a in range(small.size):
                b = int(idx[a, 1])
                if (1.0 - dist[a, 1] >= thr and int(idx[b, 1]) == a
                        and small[a] < small[b]):
                    labels[labels == small[b]] = small[a]
                    merged = True
            if not merged:
                break
            _, labels = np.unique(labels, return_inverse=True)
    return labels


def promote_noise(labels: np.ndarray, scores: np.ndarray, count: int) -> np.ndarray:
    """Give the `count` least-supported points their own singleton clusters.

    Singletons contribute no pairs, so this is an abstention that trades NMI
    (higher predicted entropy) for ARI (a smaller pair-count denominator).
    """
    if count <= 0:
        return labels
    pick = np.argpartition(-scores, count - 1)[:count]
    pick = pick[scores[pick] >= 0]
    labels = labels.copy()
    labels[pick] = np.arange(NOISE_ID_BASE, NOISE_ID_BASE + pick.size)
    return labels


def fallback(docs: list[str], mask: list[bool]) -> list[int]:
    try:
        feats = lsa_features(docs)
        if feats is None:
            raise ValueError
        k = max(2, min(60, len(docs) // 40))
        labels = MiniBatchKMeans(k, random_state=0, n_init=3).fit_predict(feats)
    except Exception:  # noqa: BLE001
        labels = np.zeros(len(docs), np.int64)
    out, pos = [], 0
    for empty in mask:
        if empty:
            out.append(-1)
        else:
            out.append(int(labels[pos]))
            pos += 1
    return out


def looks_like_titles(raw: list[str], docs: list[str]) -> tuple[bool, float]:
    lengths = [len(t) for t in docs]
    long_enough = [x for x in lengths if x >= 40]
    median_len = float(np.median(long_enough if len(long_enough) >= 50 else lengths))
    newline_frac = float(np.mean(["\n" in t for t in raw]))
    url_frac = float(np.mean(["http" in t.lower() or "www." in t.lower() for t in raw]))
    is_titles = median_len < 150 and newline_frac < 0.02 and url_frac < 0.05
    return is_titles, median_len


def cluster_titles(texts: list[str]) -> list[int] | None:
    """The champion's arXiv path: BM25-LSA + document PPMI, heavy kNN smoothing,
    average-linkage cut at 80, then 40% of points promoted to singletons."""
    try:
        n = len(texts)
        cleaned = [clean_rich(t) for t in texts]
        blocks = []
        lsa = bm25_features(cleaned)
        if lsa is not None:
            blocks.append(normalize(lsa) * 0.5)
        doc_ppmi = ppmi_doc_features(texts)
        if doc_ppmi is not None:
            blocks.append(normalize(doc_ppmi) * 1.0)
        if not blocks:
            return None
        Z = normalize(np.hstack(blocks)) if len(blocks) > 1 else blocks[0]
        Z = knn_smooth(Z, k=20, alpha=0.6, iters=4)
        _, knn_dist = relative_density(Z)
        empty = np.fromiter((len(preprocess(t)) == 0 for t in texts), bool, n)
        labels = fcluster(linkage(Z, "average", "cosine"),
                          min(TITLE_CLUSTERS, n - 1), "maxclust").astype(np.int64)
        labels = promote_noise(labels, knn_dist[:, -1].copy(), int(n * TITLE_NOISE))
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.astype(np.int64)
        if empty.any():
            labels[empty] = -1
        return [int(x) for x in labels]
    except Exception:  # noqa: BLE001
        return None


def cluster_texts(texts: list[str]) -> list[int]:
    start = time.perf_counter()
    n_total = len(texts)
    if n_total < 3:
        return list(range(n_total))

    processed = [preprocess(t) for t in texts]
    empty_mask = [len(t) == 0 for t in processed]
    docs = [t for t, empty in zip(processed, empty_mask) if not empty]
    if len(docs) < 3:
        return [0] * n_total

    try:
        n = len(docs)
        is_titles, median_len = looks_like_titles(texts, docs)
        if is_titles:
            result = cluster_titles(texts)
            if result is not None:
                return result

        lsa = lsa_features(docs)
        aniso = anisotropy(lsa) if lsa is not None else 0.0
        high_aniso = aniso > ANISOTROPY_HI
        mid_length = 230 < median_len < 330

        blocks = []
        ppmi = ppmi_window_features(docs, sif=False)
        if ppmi is not None:
            if high_aniso:
                drop = 3
            elif aniso < ANISOTROPY_LO:
                drop = 1
            else:
                drop = 3 if mid_length else 2
            blocks.append(normalize(remove_top_components(ppmi, drop)) * 1.4)
        if lsa is not None:
            weight = 0.8 if aniso < ANISOTROPY_LO else 0.94
            blocks.append(normalize(lsa) * weight)
        if not blocks:
            return fallback(docs, empty_mask)

        Z = normalize(np.hstack(blocks)) if len(blocks) > 1 else blocks[0]
        if time.perf_counter() - start < TIME_BUDGET * 0.45:
            Z = knn_smooth(Z, k=12, alpha=0.2 if mid_length else 0.1, iters=1)
        if time.perf_counter() - start < TIME_BUDGET * 0.55:
            if mid_length:
                dim = 36 if high_aniso else 16 if aniso < ANISOTROPY_LO else 32
            else:
                dim = 28 if high_aniso else 12 if aniso < ANISOTROPY_LO else 32
            spectral = spectral_features(Z, dim=dim)
            if spectral is not None:
                Z = normalize(np.hstack([Z, spectral]))
        if time.perf_counter() - start > TIME_BUDGET:
            return fallback(docs, empty_mask)

        rel, knn_dist = relative_density(Z, k=16 if mid_length else 12)
        n_clusters, noise_frac = plan_from_density(rel)
        if mid_length:
            n_clusters, noise_frac = 28, 0.2

        labels = fcluster(linkage(Z, "average", "cosine"),
                          min(n_clusters, n - 1), "maxclust").astype(np.int64)
        sids = np.fromiter((script_id(t) for t in docs), np.int16, n)
        labels = merge_by_script(labels, sids)
        if aniso < ANISOTROPY_LO:
            labels = merge_closest_pair(docs, labels)

        if noise_frac > 0:
            count = int(n * noise_frac)
            if count > 0:
                scores = knn_dist[:, -1].copy()
                scores[sids > 0] = -1.0
                if aniso < ANISOTROPY_LO:
                    scores = protect_cohesive(scores, labels, Z, n)
                labels = promote_noise(labels, scores, count)

        if not mid_length:
            labels = absorb_small(labels, Z, 0.77)
            _, labels = np.unique(labels, return_inverse=True)
            labels = merge_mutual_nn(labels, Z)

        _, labels = np.unique(labels, return_inverse=True)
        out, pos = [], 0
        for empty in empty_mask:
            if empty:
                out.append(-1)
            else:
                out.append(int(labels[pos]))
                pos += 1
        return out
    except Exception:  # noqa: BLE001
        return fallback(docs, empty_mask)
