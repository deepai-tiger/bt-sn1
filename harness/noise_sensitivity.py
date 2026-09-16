"""Check the noise-mode decision against each subset's true noise share.

Reclaiming rejects beats bucketing them by about 0.05 averaged over the round
sets, but that average is taken at 17-20% ground-truth noise and the platform
reports subsets from 13% to 29%. Since this exact decision was already wrong
once -- it was tuned on rounds carrying 30-42% noise, where bucketing wins --
the thing worth knowing is not the mean but whether the ranking holds across
the range, and where it would flip.

    /tmp/venv/bin/python harness/noise_sensitivity.py
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

SUBMISSION = Path(__file__).parent.parent / "champion-code" / "code_submission_v1.py"


def load_submission():
    spec = importlib.util.spec_from_file_location("submission_under_test", SUBMISSION)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_under_test"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds-dir", action="append", type=Path, default=None)
    args = parser.parse_args()
    dirs = args.rounds_dir or [Path("/tmp/sn1_rounds"), Path("/tmp/sn1_holdout")]

    module = load_submission()
    modes = ("none", "bucket", "singleton")

    print(f"{'subset':<28}{'gt noise':>10}{'gt k':>6}"
          + "".join(f"{m:>11}" for m in modes) + f"{'best':>11}")
    rows: list[tuple[float, dict[str, float]]] = []
    for directory in dirs:
        for path in sorted(directory.glob("round_*_subset_*.json")):
            payload = json.loads(path.read_text())
            truth = np.asarray(payload["labels"], np.int64)
            noise_frac = float((truth < 0).mean())
            true_k = int(len(np.unique(truth[truth >= 0])))

            scored: dict[str, float] = {}
            for mode in modes:
                module.configure(noise_mode=mode, title_noise_mode=mode)
                labels = module.cluster_texts(list(payload["texts"]))
                scored[mode] = score_subset(truth, np.asarray(labels))["combined"]
            rows.append((noise_frac, scored))
            best = max(scored, key=scored.get)
            print(f"{payload['name']:<28}{noise_frac:>9.1%}{true_k:>6}"
                  + "".join(f"{scored[m]:>11.4f}" for m in modes)
                  + f"{best:>11}", flush=True)

    print("\nby ground-truth noise share:")
    bands = [(0.0, 0.18), (0.18, 0.24), (0.24, 1.0)]
    for low, high in bands:
        band = [s for frac, s in rows if low <= frac < high]
        if not band:
            continue
        means = {m: float(np.mean([s[m] for s in band])) for m in modes}
        best = max(means, key=means.get)
        print(f"  {low:.0%}-{high:.0%}  n={len(band):<3}"
              + "".join(f"{means[m]:>11.4f}" for m in modes)
              + f"   best: {best}")


if __name__ == "__main__":
    main()
