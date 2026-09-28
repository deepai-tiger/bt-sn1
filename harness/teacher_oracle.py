"""What is a better distilled embedding actually worth?

The distilled blob is the obvious thing to invest in, and on the held-out
rounds the whole of it is worth +0.008: 0.3543 with it against 0.3461 with
`w_distilled = 0`. Retraining it against the correct teacher moved that number
by less than the noise between two round sets.

Before spending more on it, this replaces the blob's output with the teacher's
*actual* embeddings for the same texts -- a blob of infinite fidelity, which no
amount of distillation can beat -- and sweeps its weight. That bounds the whole
line of work: if a perfect blob is worth little, blob fidelity is not what is
holding the score down, and if it is worth a lot, the distillation is where to
spend.

Needs the round texts encoded by the teacher (`SN1_TEACHER_DIR`).

    /tmp/venv/bin/python harness/teacher_oracle.py --rounds-dir /tmp/mp_holdout
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
from score import pred_stats, score_subset  # noqa: E402

SUBMISSION = Path(__file__).parent.parent / "champion-code" / "code_submission_v1.py"
WEIGHTS = (0.0, 8.0)


def teacher_table(teacher_dir: Path) -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {}
    for name in ("reddit", "tweets", "arxiv"):
        path = teacher_dir / f"{name}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as data:
            for text, vector in zip(data["texts"], data["embeddings"]):
                table[str(text)] = vector
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds-dir", type=Path, default=Path("/tmp/mp_holdout"))
    parser.add_argument("--teacher-dir", type=Path,
                        default=Path(os.environ.get("SN1_TEACHER_DIR",
                                                    "/tmp/sn1_mpnet")))
    parser.add_argument("--submission", type=Path, default=SUBMISSION)
    parser.add_argument("--subset", action="append", default=None)
    parser.add_argument("--set", action="append", default=None,
                        metavar="KEY=JSON", help="extra CONFIG override")
    args = parser.parse_args()

    extra: dict = {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        extra[key] = json.loads(value)

    print(f"loading teacher embeddings from {args.teacher_dir}", flush=True)
    table = teacher_table(args.teacher_dir)
    print(f"  {len(table)} texts", flush=True)

    module = load_submission(args.submission)
    files = sorted(args.rounds_dir.glob("round_*_subset_*.json"))
    if args.subset:
        files = [f for f in files if any(s in f.stem for s in args.subset)]

    payloads = [json.loads(f.read_text()) for f in files]
    covered = [sum(t in table for t in p["texts"]) / len(p["texts"])
               for p in payloads]
    print(f"teacher coverage of round texts: {np.mean(covered):.1%}")

    # The submission asks for the blob's prediction through this one function,
    # so swapping the teacher's real vectors in here is enough to simulate a
    # blob that is exactly right.
    original = module.distilled_features
    # The featurizer is handed *cleaned* text, not the round's raw text, so the
    # lookup has to be keyed the same way or every row misses and the fallback
    # hides it. (It did exactly that on the first run: the teacher rows came
    # back identical to the blob rows.)
    keyed = {module.clean_text(text): vector for text, vector in table.items()}
    seen: dict[str, float] = {}

    def perfect(docs: list[str]) -> np.ndarray | None:
        hits = sum(1 for d in docs if d in keyed)
        seen["coverage"] = hits / max(1, len(docs))
        if hits < len(docs) * 0.5:
            raise SystemExit(f"teacher covers only {seen['coverage']:.1%} of "
                             "the cleaned docs; the key is still wrong")
        width = len(next(iter(keyed.values())))
        out = np.zeros((len(docs), width), np.float32)
        for row, doc in enumerate(docs):
            vector = keyed.get(doc)
            if vector is not None:
                out[row] = vector
        return out

    base = copy.deepcopy(module.CONFIG)
    print(f"\n{'w_distilled':>12}{'mean':>9}{'ari':>8}{'nmi':>8}{'k':>7}"
          f"{'source':>10}")
    for label, patched in (("blob", False), ("teacher", True)):
        module.distilled_features = perfect if patched else original
        for weight in WEIGHTS:
            if weight == 0.0 and patched:
                continue  # identical to the blob row at zero weight
            module.CONFIG.clear()
            module.CONFIG.update(base)
            module.CONFIG.update({"noise_mode": "singleton",
                                  "title_noise_mode": "singleton",
                                  "w_distilled": weight,
                                  "title_w_distilled": weight})
            module.CONFIG.update(extra)
            rows = []
            for payload in payloads:
                pred = module.cluster_texts(list(payload["texts"]))
                rows.append({**score_subset(payload["labels"], pred),
                             **pred_stats(pred)})
            n = len(rows)
            print(f"{weight:>12.1f}{sum(r['combined'] for r in rows) / n:>9.4f}"
                  f"{sum(r['ari'] for r in rows) / n:>8.4f}"
                  f"{sum(r['nmi'] for r in rows) / n:>8.4f}"
                  f"{sum(r['num_clusters'] for r in rows) / n:>7.0f}"
                  f"{label:>10}", flush=True)

    module.distilled_features = original
    module.CONFIG.clear()
    module.CONFIG.update(base)


if __name__ == "__main__":
    main()
