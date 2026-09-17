"""Grid-search CONFIG overrides against the local rounds.

`sweep.py` answers one prepared question at a time and prints a table per
experiment. This runs an arbitrary list of override dicts, which is what
exploration needs: the arXiv path in particular is cheap enough (~2s a subset)
that a hundred combinations is a few minutes.

    /tmp/venv/bin/python harness/grid.py --subset arxiv --spec specs/titles.json

A spec file is a JSON object mapping a variant name to CONFIG overrides.
"""

from __future__ import annotations

import argparse
import copy
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


def load_submission(path: Path, blob: Path | None = None):
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_under_test"] = module
    spec.loader.exec_module(module)
    if blob is not None:
        # Inject a candidate distilled embedding without editing the source, so
        # a blob can be A/B'd before it is baked in.
        module.DISTILLED_BLOB = blob.read_text(encoding="utf-8").strip()
        module._DISTILLED_CACHE = None
        if module.load_distilled() is None:
            raise SystemExit(f"{blob} did not decode")
        matrix, word, char = module.load_distilled()
        print(f"blob: {matrix.shape} over {word}+{char} buckets "
              f"({len(module.DISTILLED_BLOB):,} characters)")
    return module


def run(module, rounds: list[dict], overrides: dict) -> dict:
    base = copy.deepcopy(module.CONFIG)
    unknown = set(overrides) - set(module.CONFIG)
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    module.CONFIG.update(overrides)
    try:
        rows = []
        for payload in rounds:
            started = time.perf_counter()
            pred = module.cluster_texts(list(payload["texts"]))
            rows.append({"seconds": time.perf_counter() - started,
                         **score_subset(payload["labels"], pred),
                         **pred_stats(pred)})
    finally:
        module.CONFIG.clear()
        module.CONFIG.update(base)
    n = len(rows)
    return {
        "mean": sum(r["combined"] for r in rows) / n,
        "ari": sum(r["ari"] for r in rows) / n,
        "nmi": sum(r["nmi"] for r in rows) / n,
        "max_seconds": max(r["seconds"] for r in rows),
        "clusters": sum(r["num_clusters"] for r in rows) / n,
        "size_median": sum(r["size_median"] for r in rows) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--subset", action="append", default=None)
    parser.add_argument("--rounds-dir", type=Path, default=ROUNDS_DIR)
    parser.add_argument("--submission", type=Path, default=SUBMISSION)
    parser.add_argument("--blob", type=Path, default=None,
                        help="candidate distilled-embedding blob to inject")
    parser.add_argument("--base", action="append", default=None,
                        metavar="KEY=JSON", help="override applied to every variant")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    variants: dict[str, dict] = json.loads(args.spec.read_text())
    module = load_submission(args.submission, args.blob)
    if args.base:
        for item in args.base:
            key, _, value = item.partition("=")
            if key not in module.CONFIG:
                raise SystemExit(f"unknown config key {key!r}")
            module.CONFIG[key] = json.loads(value)
        print(f"base overrides: {args.base}")

    files = sorted(args.rounds_dir.glob("round_*_subset_*.json"))
    if args.subset:
        files = [f for f in files if any(s in f.stem for s in args.subset)]
    if not files:
        raise SystemExit(f"no rounds matched in {args.rounds_dir}")
    rounds = [json.loads(f.read_text()) for f in files]
    print(f"{len(rounds)} subsets, {len(variants)} variants\n")
    print(f"{'variant':<34} {'mean':>7} {'ari':>7} {'nmi':>7} {'secs':>6} "
          f"{'k':>6} {'med':>5}")
    print("-" * 78)

    out = {}
    for name, overrides in variants.items():
        result = run(module, rounds, overrides)
        out[name] = result
        print(f"{name:<34} {result['mean']:>7.4f} {result['ari']:>7.4f} "
              f"{result['nmi']:>7.4f} {result['max_seconds']:>6.1f} "
              f"{result['clusters']:>6.0f} {result['size_median']:>5.0f}",
              flush=True)

    print("-" * 78)
    best = max(out, key=lambda v: out[v]["mean"])
    print(f"best: {best} ({out[best]['mean']:.4f})")
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
