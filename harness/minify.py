"""Produce a submission-ready copy of the solution under the character limit.

The readable source is the source of truth. The platform caps a submission at
50,000 *characters*, and a distilled-embedding blob wants roughly 30,000 of
them, so the logic has to be compressed before a blob will fit.

`python-minifier` renames locals, strips annotations, docstrings and comments,
and collapses statements. On this file it buys roughly 3x, which is how the
previous submission fitted 18k characters of logic alongside two large blobs.

    pip install python-minifier
    /tmp/venv/bin/python harness/minify.py

Always re-score the minified output before submitting -- renaming is only
supposed to be semantics-preserving:

    /tmp/venv/bin/python harness/evaluate.py build/submission.min.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LIMIT = 50_000
DEFAULT_SOURCE = Path(__file__).parent.parent / "champion-code" / "code_submission_v1.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, nargs="?", default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent.parent / "build" / "submission.min.py")
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    print(f"source: {len(source):,} characters")

    try:
        import python_minifier
        from python_minifier import RemoveAnnotationsOptions
    except ImportError:
        raise SystemExit("python-minifier is not installed: pip install python-minifier")

    minified = python_minifier.minify(
        source,
        # Not `remove_annotations=True`. That also strips *class attribute*
        # annotations, and it does it by rewriting `texts: list[str]` to
        # `texts: 0` -- which leaves valid Python that pydantic then rejects at
        # class-creation time, so the whole submission fails on import. The
        # options object defaults to keeping class attribute annotations, which
        # is exactly what the request/response models need.
        remove_annotations=RemoveAnnotationsOptions(
            remove_variable_annotations=True,
            remove_return_annotations=True,
            remove_argument_annotations=True,
            remove_class_attribute_annotations=False,
        ),
        remove_pass=True,
        remove_literal_statements=True,   # drops docstrings
        combine_imports=True,
        hoist_literals=True,
        rename_locals=True,
        rename_globals=True,
        # `cluster_texts` and `make_app` are the platform's entry points;
        # CONFIG is what the harness overrides; the two distilled-embedding
        # names let the build be verified and A/B'd like the source can be.
        # Together they cost about 60 characters of the 50,000.
        preserve_globals=["cluster_texts", "make_app", "CONFIG", "configure",
                          "DISTILLED_BLOB", "load_distilled"],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(minified, encoding="utf-8")

    chars = len(minified)
    headroom = LIMIT - chars
    print(f"minified: {chars:,} characters ({chars / len(source):.0%} of source)")
    print(f"written to {args.out}")
    smoke_test(args.out)
    if headroom < 0:
        print(f"OVER LIMIT by {-headroom:,} characters", file=sys.stderr)
        raise SystemExit(1)
    # 15 bits per codepoint is the packing the blob loader expects.
    print(f"headroom: {headroom:,} characters "
          f"= ~{headroom * 15 / 8 / 1024:.0f} KiB of packed blob")


def smoke_test(path: Path) -> None:
    """Import the build and exercise both entry points.

    Renaming is only *supposed* to be semantics-preserving. It silently was not
    here, and the failure was at import time in the FastAPI layer -- which no
    amount of scoring `cluster_texts` would have caught, because scoring never
    builds the app.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("submission_smoke_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_smoke_test"] = module
    spec.loader.exec_module(module)

    app = module.make_app()
    if app is None:
        raise SystemExit("make_app() returned nothing")

    labels = module.cluster_texts([f"a short document about topic {i % 7}"
                                   for i in range(120)])
    if len(labels) != 120:
        raise SystemExit(f"cluster_texts returned {len(labels)} labels for 120 texts")

    if module.DISTILLED_BLOB and module.load_distilled() is None:
        raise SystemExit("the baked blob does not decode in the minified build")
    print("smoke test: imports, builds the app, clusters, decodes the blob")


if __name__ == "__main__":
    main()
