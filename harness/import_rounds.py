"""Turn the platform's own previous-round input files into scored rounds.

Every replica round so far was assembled from public corpora that only
resemble what Gravity crawls, and the mismatch is measurable: the rebuilt
replica runs 23-30% ground-truth noise against the platform's 13-22%, and
scores both our submission and the champion reconstruction ~0.06-0.08 below
what the platform gave them. The fix is to stop approximating the texts.

`shared/competition/scripts/get_previous_round_input_files.py` in the Apex
repo lists presigned download URLs for past rounds' input parquet files (it
needs a linked hotkey, so it has to run on the miner's machine). Point this
script at the downloaded files. If a file already carries labels they are used
as-is; otherwise ground truth is baked with the same recipe as the platform --
all-mpnet-base-v2, UMAP(n_neighbors=15, n_components=5, min_dist=0, cosine),
HDBSCAN(min_cluster_size=25) -- via `make_rounds.ground_truth`.

    python3 harness/import_rounds.py downloads/*.parquet --out /tmp/platform_rounds
    python3 harness/grid.py --spec harness/specs/one.json --rounds-dir /tmp/platform_rounds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TEXT_COLUMNS = ("text", "texts", "content", "body", "post")
LABEL_COLUMNS = ("label", "labels", "cluster", "cluster_id", "ground_truth")
MODEL = "sentence-transformers/all-mpnet-base-v2"


def read_table(path: Path) -> tuple[list[str], list[int] | None]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    names = {name.lower(): name for name in table.column_names}
    text_col = next((names[c] for c in TEXT_COLUMNS if c in names), None)
    if text_col is None:
        raise SystemExit(f"{path.name}: no text column among {table.column_names}")
    texts = [str(t) if t is not None else "" for t in table.column(text_col).to_pylist()]
    label_col = next((names[c] for c in LABEL_COLUMNS if c in names), None)
    labels = None
    if label_col is not None:
        labels = [int(v) for v in table.column(label_col).to_pylist()]
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("/tmp/platform_rounds"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = None
    for path in args.files:
        texts, labels = read_table(path)
        source = "file"
        if labels is None:
            if model is None:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(MODEL)
            from make_rounds import describe, ground_truth

            print(f"{path.name}: encoding {len(texts)} texts", flush=True)
            embeddings = model.encode(texts, batch_size=64, show_progress_bar=False,
                                      normalize_embeddings=True)
            labels = ground_truth(np.asarray(embeddings, np.float32), args.seed).tolist()
            source = "baked"
            stats = describe(np.asarray(labels))
        else:
            from make_rounds import describe

            stats = describe(np.asarray(labels))
        # Platform keys already read `round_0054_subset_1_<hash>`, which is the
        # pattern every evaluator globs for; anything else is coerced into it.
        name = path.stem
        if not name.startswith("round_"):
            name = f"round_{name}"
        if "_subset_" not in name:
            name = f"{name}_subset_1"
        (args.out / f"{name}.json").write_text(json.dumps({
            "name": name, "texts": texts, "labels": labels,
            "label_source": source, "stats": stats}))
        print(f"{name}: {len(texts)} texts, labels {source}, {stats}", flush=True)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    main()
