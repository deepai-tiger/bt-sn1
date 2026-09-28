"""Ad-hoc probe: run one stage combination over chosen subsets and report.

The sweep harness reloads the submission and runs full rounds, which is the
right tool for A/B-ing a committed default. This one is for exploration: it
caches the expensive feature/manifold stages so clustering choices can be
compared in seconds instead of minutes.

    /tmp/venv/bin/python harness/probe.py --subset arxiv --grid mcs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).parent))
from score import pred_stats, score_subset  # noqa: E402

ROUNDS_DIR = Path("/tmp/sn1_rounds")
SUBMISSION = Path(__file__).parent.parent / "champion-code" / "code_submission_v1.py"


def load_submission():
    spec = importlib.util.spec_from_file_location("submission_under_test", SUBMISSION)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_under_test"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", action="append", default=None)
    parser.add_argument("--set", action="append", default=None, metavar="KEY=JSON")
    parser.add_argument("--label", default="variant")
    args = parser.parse_args()

    module = load_submission()
    if args.set:
        for item in args.set:
            key, _, value = item.partition("=")
            if key not in module.CONFIG:
                raise SystemExit(f"unknown config key {key!r}")
            module.CONFIG[key] = json.loads(value)

    files = sorted(ROUNDS_DIR.glob("round_*_subset_*.json"))
    if args.subset:
        files = [f for f in files if any(s in f.stem for s in args.subset)]

    rows = []
    for path in files:
        payload = json.loads(path.read_text())
        started = time.perf_counter()
        pred = module.cluster_texts(list(payload["texts"]))
        elapsed = time.perf_counter() - started
        result = score_subset(payload["labels"], pred)
        stats = pred_stats(pred)
        rows.append(result["combined"])
        print(f"  {payload['name']:<26} ari={result['ari']:.4f} nmi={result['nmi']:.4f} "
              f"combined={result['combined']:.4f} {elapsed:5.1f}s "
              f"k={stats['num_clusters']:<5} med={stats['size_median']}")
    print(f"{args.label}: MEAN {sum(rows) / len(rows):.4f}")


if __name__ == "__main__":
    main()
