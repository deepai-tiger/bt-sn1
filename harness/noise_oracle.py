"""Is the noise decision limited by the shape we choose, or by knowing which
points are noise?

The three output shapes have very different ceilings against a perfect
detector (bucket 1.000, singleton 0.827, reclaim 0.810), so whether flooding
noise is worth anything depends entirely on whether our own outlier ranking
can find the ground truth's noise. This measures that directly:

* how well the clusterer's own rejects, and the outlier ranking, agree with
  the ground truth's noise set, and
* what the score would be if the noise set were known exactly, holding the
  rest of our prediction fixed.

The gap between the oracle row and the shipped row is the prize available for
better noise detection; the gap to 1.000 is the prize for better features.

    /tmp/venv/bin/python harness/noise_oracle.py --subset subset_1
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

    print(f"{'subset':<26}{'shipped':>9}{'orc-buck':>10}{'orc-sing':>10}"
          f"{'rej%':>7}{'prec':>7}{'rec':>7}{'gt%':>7}")
    rows = []
    for path in files:
        payload = json.loads(path.read_text())
        truth = np.asarray(payload["labels"])
        noise = truth < 0

        shipped = np.asarray(module.cluster_texts(list(payload["texts"])))

        # The same run with noise handling disabled, to see the raw reject set.
        base = copy.deepcopy(module.CONFIG)
        module.CONFIG.update({"noise_mode": "singleton",
                              "title_noise_mode": "singleton",
                              "extra_noise_frac": 0.0,
                              "title_extra_noise_frac": 0.0})
        try:
            split = np.asarray(module.cluster_texts(list(payload["texts"])))
        finally:
            module.CONFIG.clear()
            module.CONFIG.update(base)

        # Singletons in that run are the points the clusterer would not place.
        ids, counts = np.unique(split, return_counts=True)
        singles = np.isin(split, ids[counts == 1])

        overlap = int((singles & noise).sum())
        precision = overlap / max(1, int(singles.sum()))
        recall = overlap / max(1, int(noise.sum()))

        # Oracle: keep our clusters, but label exactly the true noise set.
        bucket = shipped.copy()
        bucket[noise] = shipped.max() + 1
        singleton = shipped.copy()
        singleton[noise] = np.arange(shipped.max() + 1,
                                     shipped.max() + 1 + int(noise.sum()))

        row = (score_subset(truth, shipped)["combined"],
               score_subset(truth, bucket)["combined"],
               score_subset(truth, singleton)["combined"],
               float(singles.mean()), precision, recall, float(noise.mean()))
        rows.append(row)
        print(f"{path.stem:<26}{row[0]:>9.4f}{row[1]:>10.4f}{row[2]:>10.4f}"
              f"{row[3]:>7.1%}{row[4]:>7.1%}{row[5]:>7.1%}{row[6]:>7.1%}",
              flush=True)

    mean = np.asarray(rows).mean(0)
    print(f"{'MEAN':<26}{mean[0]:>9.4f}{mean[1]:>10.4f}{mean[2]:>10.4f}"
          f"{mean[3]:>7.1%}{mean[4]:>7.1%}{mean[5]:>7.1%}{mean[6]:>7.1%}")
    print(f"\nperfect noise detection is worth {mean[1] - mean[0]:+.4f} "
          f"(bucket) / {mean[2] - mean[0]:+.4f} (singleton) over what we ship")


if __name__ == "__main__":
    main()
