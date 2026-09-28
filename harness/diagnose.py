"""Locate the bottleneck: is the feature space or the clusterer the limit?

Every hyperparameter on the social path measures within +/-0.005 of the
default, which says the search is done and something structural is binding.
This separates the two candidates.

The informative number is the **oracle**: each point is assigned to whichever
ground-truth cluster has the nearest centroid *in our own feature space*. That
is the best any clusterer could do with these features if it were handed the
true cluster locations for free. If the oracle is far above the pipeline, the
clusterer is leaving value on the table; if it is close, the features are the
ceiling and no amount of clustering work will help.

    /tmp/venv/bin/python harness/diagnose.py --subset subset_1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).parent))
from score import score_subset  # noqa: E402

ROUNDS_DIR = Path("/tmp/sn1_rounds")
SUBMISSION = Path(__file__).parent.parent / "champion-code" / "code_submission_v1.py"


def load_submission():
    spec = importlib.util.spec_from_file_location("submission_under_test", SUBMISSION)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_under_test"] = module
    spec.loader.exec_module(module)
    return module


def oracle_labels(E: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Nearest true-centroid assignment in the submission's own space."""
    from sklearn.preprocessing import normalize

    ids = np.unique(truth)
    centres = normalize(np.stack([E[truth == i].mean(0) for i in ids]))
    return ids[(E @ centres.T).argmax(1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", action="append", default=None)
    args = parser.parse_args()

    from sklearn.cluster import AgglomerativeClustering, KMeans

    module = load_submission()
    files = sorted(ROUNDS_DIR.glob("round_*_subset_*.json"))
    if args.subset:
        files = [f for f in files if any(s in f.stem for s in args.subset)]

    for path in files:
        payload = json.loads(path.read_text())
        texts = list(payload["texts"])
        truth = np.asarray(payload["labels"], np.int64)

        cleaned = [module.clean_text(t) for t in texts]
        keep = [i for i, t in enumerate(cleaned) if t]
        docs = [cleaned[i] for i in keep]
        truth_kept = truth[keep]
        is_titles, _ = module.detect_titles(texts, docs)

        clock = module.Clock(float(module.CONFIG["time_budget"]))
        Z = module.build_features(docs, is_titles, clock)
        Z = module.knn_smooth(Z, int(module.opt("smooth_k", is_titles)),
                              float(module.opt("smooth_alpha", is_titles)),
                              int(module.opt("smooth_iters", is_titles)),
                              bool(module.opt("fuzzy_smooth", is_titles)))
        E = module.reduce_space(Z, is_titles)

        true_k = int(len(np.unique(truth_kept[truth_kept >= 0])))
        rows = {}
        rows["pipeline"] = module.cluster_texts(texts)
        rows[f"oracle(k={true_k}+noise)"] = oracle_labels(E, truth_kept)
        rows[f"kmeans(k={true_k})"] = KMeans(true_k, n_init=4, random_state=0) \
            .fit_predict(E)
        rows[f"ward(k={true_k})"] = AgglomerativeClustering(
            n_clusters=true_k).fit_predict(E)

        print(f"\n{payload['name']}  (ground truth: {true_k} clusters, "
              f"{int((truth_kept < 0).sum())} noise of {len(truth_kept)})")
        for name, labels in rows.items():
            labels = np.asarray(labels)
            reference = truth if labels.size == truth.size else truth_kept
            result = score_subset(reference, labels)
            print(f"  {name:<24} ari={result['ari']:.4f} nmi={result['nmi']:.4f} "
                  f"combined={result['combined']:.4f}")


if __name__ == "__main__":
    main()
