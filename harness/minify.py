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
    except ImportError:
        raise SystemExit("python-minifier is not installed: pip install python-minifier")

    minified = python_minifier.minify(
        source,
        remove_annotations=True,
        remove_pass=True,
        remove_literal_statements=True,   # drops docstrings
        combine_imports=True,
        hoist_literals=True,
        rename_locals=True,
        rename_globals=True,
        # `cluster_texts` and `make_app` are the platform's entry points, and
        # CONFIG is what the harness overrides.
        preserve_globals=["cluster_texts", "make_app", "CONFIG", "configure"],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(minified, encoding="utf-8")

    chars = len(minified)
    headroom = LIMIT - chars
    print(f"minified: {chars:,} characters ({chars / len(source):.0%} of source)")
    print(f"written to {args.out}")
    if headroom < 0:
        print(f"OVER LIMIT by {-headroom:,} characters", file=sys.stderr)
        raise SystemExit(1)
    # 15 bits per codepoint is the packing the blob loader expects.
    print(f"headroom: {headroom:,} characters "
          f"= ~{headroom * 15 / 8 / 1024:.0f} KiB of packed blob")


if __name__ == "__main__":
    main()
