"""How precise is our noise ranking, and where should the bucket cut fall?

`noise_oracle.py` shows that labelling the ground truth's noise set exactly is
worth +0.43, far more than any other lever we have. But a bucket only pays
while it is mostly right: its pairs grow as the square of its size, so a bucket
at 40% precision contributes four false pairs for every true one.

That makes the useful question not "which shape" but "how far down our own
ranking does precision hold". This walks the ranking and reports, at each
depth, the precision against the true noise set and the score from bucketing
exactly that prefix -- with everything else reclaimed, so the bucket is the
only thing under test.

    /tmp/venv/bin/python harness/noise_pr.py --subset arxiv
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).parent))
from grid import load_submission  # noqa: E402
from score import score_subset  # noqa: E402

ROUNDS_DIR = Path("/tmp/sn1_rounds")
SUBMISSION = Path(__file__).parent.parent / "champion-code" / "code_submission_v1.py"
DEPTHS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds-dir", type=Path, default=ROUNDS_DIR)
    parser.add_argument("--submission", type=Path, default=SUBMISSION)
    parser.add_argument("--subset", action="append", default=None)
    args = parser.parse_args()

    module = load_submission(args.submission)
    files = sorted(args.rounds_dir.glob("round_*_subset_*.json"))
    if args.subset:
        files = [f for f in files if any(s in f.stem for s in args.subset)]

    # The ranking is built inside assign_noise, which is also the only place
    # the final embedding is in scope. Capture its arguments rather than
    # rebuilding the pipeline around it.
    captured: dict = {}
    original = module.assign_noise

    def spy(labels, Z, sids, is_titles=False):
        captured["labels"] = np.asarray(labels).copy()
        captured["Z"] = np.asarray(Z).copy()
        captured["is_titles"] = is_titles
        return original(labels, Z, sids, is_titles)

    module.assign_noise = spy

    base = copy.deepcopy(module.CONFIG)
    module.CONFIG.update({"noise_mode": "none", "title_noise_mode": "none",
                          "extra_noise_frac": 0.0,
                          "title_extra_noise_frac": 0.0})

    precisions: list[list[float]] = []
    scores: list[list[float]] = []
    shipped: list[float] = []
    try:
        for path in files:
            payload = json.loads(path.read_text())
            truth = np.asarray(payload["labels"])
            noise = truth < 0

            reclaimed = np.asarray(module.cluster_texts(list(payload["texts"])))
            shipped.append(score_subset(truth, reclaimed)["combined"])

            ranking = module.outlier_score(captured["Z"], captured["labels"])
            order = np.argsort(-ranking, kind="stable")

            prow, srow = [], []
            for depth in DEPTHS:
                count = int(len(order) * depth)
                head = order[:count]
                prow.append(float(noise[head].mean()))
                trial = reclaimed.copy()
                trial[head] = reclaimed.max() + 1
                srow.append(score_subset(truth, trial)["combined"])
            precisions.append(prow)
            scores.append(srow)
            print(f"{path.stem}: reclaim {shipped[-1]:.4f}, gt noise "
                  f"{noise.mean():.1%}", flush=True)
    finally:
        module.CONFIG.clear()
        module.CONFIG.update(base)
        module.assign_noise = original

    p = np.asarray(precisions).mean(0)
    s = np.asarray(scores).mean(0)
    print(f"\nreclaim-everything baseline: {np.mean(shipped):.4f}")
    print(f"\n{'depth':>7}{'precision':>11}{'bucket score':>14}{'delta':>9}")
    for depth, pi, si in zip(DEPTHS, p, s):
        print(f"{depth:>7.0%}{pi:>11.1%}{si:>14.4f}{si - np.mean(shipped):>9.4f}")


if __name__ == "__main__":
    main()
