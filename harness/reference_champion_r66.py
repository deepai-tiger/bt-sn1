"""Reconstruction of the round-66 winner (0.4735), for A/B runs on the replica.

Source: `champion-code/round-66/top/code_submission_v0.py`, a CLI panel export
cropped at 205 columns, so both `lzma` blobs and fourteen long lines are cut.
What survives shows the same skeleton as `reference_champion.py` (round 52)
with these changes, all of which are reproduced here:

* PPMI window 6 (was 10), PPMI block weight 1.0 (was 1.4);
* top-component drops of 4 / 1 / 3 / 2 for high anisotropy / low anisotropy /
  mid-length / otherwise, plus one dropped LSA component at high anisotropy;
* a distilled-teacher block (`Ay`, weight 1.0) on the social path and a
  separate title teacher (`BP`, weight 3.0) on the title path;
* mid-length batches (median cleaned length in (230, 330)) cut average linkage
  at 28 clusters and make exactly 25% of points singletons, with no reclaim.
  Round 66 confirms the platform's social subsets take this branch: the
  winner emitted exactly int(0.25 * n) singletons on all three;
* the title path cuts at 100 clusters and makes 30% of points singletons.

The blobs are unrecoverable, so both teacher blocks use *our* distilled map.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
import reference_champion as base  # noqa: E402

_SUBMISSION = Path(__file__).resolve().parent.parent / "champion-code" / "code_submission_v1.py"


def _load_teacher():
    spec = importlib.util.spec_from_file_location("_r66_teacher_source", _SUBMISSION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TEACHER = _load_teacher()

TIME_BUDGET = 70.0
PPMI_WINDOW = 6
PPMI_DIM = 128
TEACHER_WEIGHT = 1.0
TITLE_TEACHER_WEIGHT = 3.0
ANISOTROPY_HI = 0.045
ANISOTROPY_LO = 0.03
CLUSTERS_RANGE = (40, 220)
NOISE_RANGE = (0.25, 0.40)
REL_RANGE = (0.10, 0.32)
MID_CLUSTERS = 28
MID_NOISE = 0.25
TITLE_CLUSTERS = 100
TITLE_NOISE = 0.30


def teacher(raw: list[str]) -> np.ndarray | None:
    return _TEACHER.distilled_features([_TEACHER.clean_text(t) for t in raw])


def ppmi_window_features(docs: list[str], window: int = PPMI_WINDOW) -> np.ndarray | None:
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
            for right in ids[pos + 1: pos + 1 + window]:
                rows += (left, right)
                cols += (right, left)
    if not rows:
        return None
    size = len(vocab)
    co = coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(size, size)).tocsr()
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
    ppmi = coo_matrix((pmi[keep], (co.row[keep], co.col[keep])), shape=(size, size)).tocsr()
    dim = min(PPMI_DIM, max(2, min(ppmi.shape) - 1))
    vecs = normalize(TruncatedSVD(n_components=dim, random_state=42).fit_transform(ppmi))
    tfidf = TfidfVectorizer(vocabulary=vocab, stop_words="english", sublinear_tf=True,
                            dtype=np.float32)
    return normalize(tfidf.fit_transform(docs) @ vecs)


def plan_from_density(rel: float) -> tuple[int, float]:
    frac = float(np.clip((rel - REL_RANGE[0]) / (REL_RANGE[1] - REL_RANGE[0]), 0.0, 1.0))
    return (int(round(CLUSTERS_RANGE[0] + (CLUSTERS_RANGE[1] - CLUSTERS_RANGE[0]) * frac)),
            float(NOISE_RANGE[0] + (NOISE_RANGE[1] - NOISE_RANGE[0]) * frac))


def cluster_titles(texts: list[str]) -> list[int] | None:
    try:
        n = len(texts)
        cleaned = [base.clean_rich(t) for t in texts]
        blocks = []
        lsa = base.bm25_features(cleaned, char_weight=0.35, char_features=8000)
        if lsa is not None:
            blocks.append(normalize(lsa))
        doc_ppmi = base.ppmi_doc_features(texts)
        if doc_ppmi is not None:
            blocks.append(normalize(doc_ppmi))
        dist = teacher(texts)
        if dist is not None:
            blocks.append(normalize(dist) * TITLE_TEACHER_WEIGHT)
        if not blocks:
            return None
        Z = normalize(np.hstack(blocks)) if len(blocks) > 1 else blocks[0]
        Z = base.knn_smooth(Z, k=20, alpha=0.6, iters=4)
        _, knn_dist = base.relative_density(Z, k=12)
        empty = np.fromiter((len(base.preprocess(t)) == 0 for t in texts), bool, n)
        labels = fcluster(linkage(Z, "average", "cosine"),
                          min(TITLE_CLUSTERS, n - 1), "maxclust").astype(np.int64)
        count = int(n * TITLE_NOISE)
        if count > 0:
            pick = np.argsort(-knn_dist[:, -1])[:count]
            labels[pick] = np.arange(base.NOISE_ID_BASE, base.NOISE_ID_BASE + count)
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
    processed = [base.preprocess(t) for t in texts]
    empty_mask = [len(t) == 0 for t in processed]
    docs = [t for t, empty in zip(processed, empty_mask, strict=True) if not empty]
    raw_docs = [t for t, empty in zip(texts, empty_mask, strict=True) if not empty]
    if len(docs) < 3:
        return [0] * n_total

    try:
        n = len(docs)
        is_titles, median_len = base.looks_like_titles(texts, docs)
        if is_titles:
            result = cluster_titles(texts)
            if result is not None:
                return result

        mid = 230 < median_len < 330
        lsa = base.lsa_features(docs)
        aniso = base.anisotropy(lsa) if lsa is not None else 0.0
        high = aniso > ANISOTROPY_HI
        low = aniso < ANISOTROPY_LO

        blocks = []
        ppmi = ppmi_window_features(docs)
        if ppmi is not None:
            drop = 4 if high else 1 if low else (3 if mid else 2)
            blocks.append(normalize(base.remove_top_components(ppmi, drop)))
        if lsa is not None:
            if high:
                lsa = base.remove_top_components(lsa, 1)
            weight = 1.0 if high else 0.8 if low else 0.94
            blocks.append(normalize(lsa) * weight)
        dist = teacher(raw_docs)
        if dist is not None:
            blocks.append(normalize(dist) * TEACHER_WEIGHT)
        if not blocks:
            return base.fallback(docs, empty_mask)

        Z = normalize(np.hstack(blocks)) if len(blocks) > 1 else blocks[0]
        if time.perf_counter() - start < TIME_BUDGET * 0.45:
            Z = base.knn_smooth(Z, k=12 if mid else 8, alpha=0.2 if mid else 0.1, iters=1)
        if time.perf_counter() - start < TIME_BUDGET * 0.55:
            if mid:
                dim = 36 if high else 16 if low else 32
            else:
                dim = 28 if high else 12 if low else 32
            spectral = base.spectral_features(Z, dim=dim)
            if spectral is not None:
                Z = normalize(np.hstack([Z, spectral]))

        rel, knn_dist = base.relative_density(Z, k=20 if mid else 12)
        n_clusters, noise_frac = (MID_CLUSTERS, MID_NOISE) if mid else plan_from_density(rel)
        labels = fcluster(linkage(Z, "average", "cosine"),
                          min(n_clusters, n - 1), "maxclust").astype(np.int64)
        sids = np.fromiter((base.script_id(t) for t in docs), np.int16, n)
        labels = base.merge_by_script(labels, sids)
        if low:
            labels = base.merge_closest_pair(docs, labels)

        count = int(n * noise_frac)
        if count > 0:
            scores = knn_dist[:, -1].copy()
            scores[sids > 0] = -1.0
            if low:
                scores = base.protect_cohesive(scores, labels, Z, n)
            labels = base.promote_noise(labels, scores, count)

        if not mid:
            labels = base.absorb_small(labels, Z, 0.77)
            _, labels = np.unique(labels, return_inverse=True)
            labels = base.merge_mutual_nn(labels, Z)

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
        return base.fallback(docs, empty_mask)
