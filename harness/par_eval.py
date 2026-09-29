"""Score a submission-like module over a rounds directory, one process per core.

Each worker imports the module once. Overrides are `NAME=VALUE` pairs: a name
that exists in the module's `CONFIG` dict is set there, anything else is set
as a module attribute (so reconstruction constants like `MID_CLUSTERS` can be
swept too). Several `--variant` groups run back to back on the same rounds:

    python3 harness/par_eval.py --module harness/reference_champion_r66.py \
        --rounds-dir /tmp/sn13_rounds --variant MID_CLUSTERS=24 --variant MID_CLUSTERS=32
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from score import pred_stats, score_subset  # noqa: E402

_MODULE = None


def parse_value(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _init(path: str) -> None:
    global _MODULE
    spec = importlib.util.spec_from_file_location("par_eval_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["par_eval_module"] = module
    spec.loader.exec_module(module)
    _MODULE = module


def _apply(overrides: dict) -> dict:
    saved = {}
    config = getattr(_MODULE, "CONFIG", None)
    for key, value in overrides.items():
        if isinstance(config, dict) and key in config:
            saved[("cfg", key)] = config[key]
            config[key] = value
        else:
            saved[("attr", key)] = getattr(_MODULE, key)
            setattr(_MODULE, key, value)
    return saved


def _restore(saved: dict) -> None:
    config = getattr(_MODULE, "CONFIG", None)
    for (kind, key), value in saved.items():
        if kind == "cfg":
            config[key] = value
        else:
            setattr(_MODULE, key, value)


def _run(job):
    path, overrides = job
    payload = json.loads(Path(path).read_text())
    saved = _apply(overrides)
    try:
        start = time.perf_counter()
        pred = _MODULE.cluster_texts(list(payload["texts"]))
        seconds = time.perf_counter() - start
    finally:
        _restore(saved)
    truth = payload["labels"]
    result = score_subset(truth, pred)
    gt_noise = sum(1 for x in truth if x == -1) / len(truth)
    _, sizes = np.unique(np.asarray(pred), return_counts=True)
    shape = {"real_clusters": int((sizes > 1).sum()),
             "single_frac": float(sizes[sizes == 1].sum() / len(pred))}
    return {"name": payload["name"], "seconds": seconds, "gt_noise": gt_noise,
            **result, **pred_stats(pred), **shape}


def summarize(label: str, rows: list[dict]) -> dict:
    social = [r for r in rows if "arxiv" not in r["name"]]
    arxiv = [r for r in rows if "arxiv" in r["name"]]

    def mean(items, key):
        return sum(r[key] for r in items) / len(items) if items else float("nan")

    out = {"variant": label, "mean": mean(rows, "combined"), "social": mean(social, "combined"),
           "arxiv": mean(arxiv, "combined"), "ari": mean(rows, "ari"), "nmi": mean(rows, "nmi"),
           "secs_max": max(r["seconds"] for r in rows), "rows": rows}
    print(f"{label:<44} mean {out['mean']:.4f}  social {out['social']:.4f}  "
          f"arxiv {out['arxiv']:.4f}  ARI {out['ari']:.4f}  NMI {out['nmi']:.4f}  "
          f"max {out['secs_max']:.1f}s", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--rounds-dir", type=Path, required=True)
    parser.add_argument("--subset", action="append", default=None)
    parser.add_argument("--variant", action="append", default=None,
                        help="space-separated NAME=VALUE overrides; repeatable")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--per-subset", action="store_true")
    args = parser.parse_args()

    files = sorted(args.rounds_dir.glob("round_*_subset_*.json"))
    if args.subset:
        files = [f for f in files if any(s in f.stem for s in args.subset)]
    variants = args.variant or [""]
    parsed = []
    for spec in variants:
        overrides = {}
        for item in spec.split():
            key, _, value = item.partition("=")
            overrides[key] = parse_value(value)
        parsed.append((spec or "baseline", overrides))

    results = []
    ctx = get_context("fork")
    with ctx.Pool(args.jobs, initializer=_init, initargs=(args.module,)) as pool:
        for label, overrides in parsed:
            rows = pool.map(_run, [(str(f), overrides) for f in files], chunksize=1)
            out = summarize(label, rows)
            if args.per_subset:
                for r in rows:
                    print(f"    {r['name']:<26} {r['combined']:.4f}  ARI {r['ari']:.4f}  "
                          f"NMI {r['nmi']:.4f}  k {r['real_clusters']:4d}  "
                          f"single {r['single_frac']:.3f}  max {r['size_max']:4d}  "
                          f"gt_noise {r['gt_noise']:.3f}  {r['seconds']:.1f}s")
            results.append(out)
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
