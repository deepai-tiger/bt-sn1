"""Bake a trained distilled-embedding blob into the submission source.

The blob is 25k characters of packed binary, so it is generated (by
`distill_fit.py`), tested (`--blob` on the harness runners), and only then
written into `DISTILLED_BLOB`. Keeping that a separate step means the source
file is never hand-edited with a wall of codepoints.

    /tmp/venv/bin/python harness/embed_blob.py /tmp/sn1_distill/blob.txt
    /tmp/venv/bin/python harness/embed_blob.py --clear
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SUBMISSION = Path(__file__).parent.parent / "champion-code" / "code_submission_v1.py"
PATTERN = re.compile(r'^DISTILLED_BLOB = ".*"$', re.MULTILINE)
LIMIT = 50_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("blob", type=Path, nargs="?")
    parser.add_argument("--target", type=Path, default=SUBMISSION)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    if not args.clear and args.blob is None:
        raise SystemExit("pass a blob file or --clear")

    payload = "" if args.clear else args.blob.read_text(encoding="utf-8").strip()
    if any(ch in payload for ch in '"\\\n\r'):
        raise SystemExit("blob contains characters that cannot go in a literal")

    source = args.target.read_text(encoding="utf-8")
    if not PATTERN.search(source):
        raise SystemExit("could not find the DISTILLED_BLOB assignment")
    updated = PATTERN.sub(f'DISTILLED_BLOB = "{payload}"', source, count=1)
    args.target.write_text(updated, encoding="utf-8")

    print(f"blob: {len(payload):,} characters")
    print(f"source now {len(updated):,} characters "
          f"(minify before submitting; the {LIMIT:,} limit applies to the "
          f"minified build)")


if __name__ == "__main__":
    main()
