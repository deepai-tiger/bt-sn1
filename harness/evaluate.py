"""Score a submission file against the locally generated rounds.

    /tmp/venv/bin/python harness/evaluate.py champion-code/code_submission_v1.py

Runs with the submission venv so the measured wall time reflects the sandbox's
package versions. Imports the submission's `cluster_texts` directly rather than
going over HTTP; the FastAPI layer adds nothing worth measuring.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# Match the sandbox: the submission pins BLAS to one thread at import time.
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent))
from score import pred_stats, score_subset  # noqa: E402

ROUNDS_DIR = Path("/tmp/sn1_rounds")
TIME_LIMIT = 90.0


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_under_test"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "cluster_texts"):
        raise AttributeError(f"{path} has no cluster_texts()")
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--rounds-dir", type=Path, default=ROUNDS_DIR)
    parser.add_argument("--subset", action="append", default=None,
                        help="substring filter on subset name; repeatable")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    module = load_submission(args.submission)
    files = sorted(args.rounds_dir.glob("round_*_subset_*.json"))
    if args.subset:
        files = [f for f in files if any(s in f.stem for s in args.subset)]
    if not files:
        raise SystemExit(f"no round files in {args.rounds_dir}")

    results = []
    print(f"{'subset':<28} {'ARI':>7} {'NMI':>7} {'combined':>9} "
          f"{'secs':>6} {'clusters':>9} {'noise':>6} {'med':>5}")
    print("-" * 92)

    for path in files:
        payload = json.loads(path.read_text())
        texts, truth = payload["texts"], payload["labels"]

        start = time.perf_counter()
        pred = module.cluster_texts(list(texts))
        elapsed = time.perf_counter() - start

        result = score_subset(truth, pred)
        stats = pred_stats(pred)
        over = "  OVER LIMIT" if elapsed > TIME_LIMIT else ""
        results.append({"subset": payload["name"], "seconds": elapsed,
                        **result, **stats})
        print(f"{payload['name']:<28} {result['ari']:>7.4f} {result['nmi']:>7.4f} "
              f"{result['combined']:>9.4f} {elapsed:>6.1f} "
              f"{stats['num_clusters']:>9} {stats['noise_count']:>6} "
              f"{stats['size_median']:>5}{over}")

    print("-" * 92)
    mean = sum(r["combined"] for r in results) / len(results)
    social = [r for r in results if "arxiv" not in r["subset"]]
    arxiv = [r for r in results if "arxiv" in r["subset"]]
    print(f"MEAN combined: {mean:.4f}   (n={len(results)})")
    if social:
        print(f"  social mean: {sum(r['combined'] for r in social) / len(social):.4f}")
    if arxiv:
        print(f"  arxiv  mean: {sum(r['combined'] for r in arxiv) / len(arxiv):.4f}")
    print(f"  max seconds: {max(r['seconds'] for r in results):.1f} / {TIME_LIMIT}")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"submission": str(args.submission), "mean_combined": mean,
             "results": results}, indent=2))


if __name__ == "__main__":
    main()
