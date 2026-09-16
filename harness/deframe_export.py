"""Recover source from an Apex CLI export.

`apex submission get` prints the file inside a Rich panel: a box border, a
right-aligned line number, then the code padded out to a fixed panel width.
Code wider than the panel is **silently cut**, which is why the v1 export did
not parse at all.

This strips the frame and reports which lines were truncated, so what is
missing is explicit rather than a mystery syntax error.

    /tmp/venv/bin/python harness/deframe_export.py \\
        champion-code/code_submission_v2.py --out /tmp/v2.py
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

# "│" then a right-aligned line number then one space, and a trailing "│".
ROW = re.compile(r"^\u2502(\s*\d+)\s(.*)\u2502$")


def deframe(text: str) -> tuple[list[str], list[int], int]:
    rows: list[tuple[int, str]] = []
    widths: set[int] = set()
    for line in text.split("\n"):
        match = ROW.match(line)
        if not match:
            continue
        number = int(match.group(1))
        body = match.group(2)
        widths.add(len(body))
        rows.append((number, body.rstrip()))

    # The panel pads every row to the same width, so the padded width is the
    # most common one and anything reaching it may have been cut.
    panel = max(widths) if widths else 0
    code: list[str] = []
    suspect: list[int] = []
    expected = 1
    for number, body in rows:
        while expected < number:          # a row the panel dropped entirely
            code.append("")
            suspect.append(expected)
            expected += 1
        code.append(body)
        if len(body) >= panel:
            suspect.append(number)
        expected = number + 1
    return code, suspect, panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    code, suspect, panel = deframe(args.export.read_text(encoding="utf-8"))
    source = "\n".join(code) + "\n"
    args.out.write_text(source, encoding="utf-8")

    print(f"{len(code)} lines, panel body width {panel}")
    print(f"recovered {len(source):,} characters -> {args.out}")
    if suspect:
        print(f"{len(suspect)} line(s) at or over the panel width, so possibly "
              f"truncated: {suspect}")
    try:
        ast.parse(source)
        print("parses cleanly")
    except SyntaxError as exc:
        print(f"SyntaxError at line {exc.lineno}: {exc.msg}")
        if exc.lineno:
            print(f"  {code[exc.lineno - 1][:160]}")


if __name__ == "__main__":
    main()
