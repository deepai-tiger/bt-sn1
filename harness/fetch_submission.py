#!/usr/bin/env python3
"""Download a submission's code and evaluation files uncropped.

`apex result <id> -f <file>` prints files inside a Rich panel, which crops
every line at the terminal width. Minified submissions keep their baked
weights on one very long line, so a panel export loses them entirely (the
round-66 winner's two `lzma` blobs, for example). This makes the same API
calls the CLI and dashboard make and writes the raw text to disk instead,
following pagination to the end of each file.

Run it from the root of an Apex checkout, with a linked hotkey, the same way
as `shared/competition/scripts/get_previous_round_input_files.py`:

    uv run --group dev python /path/to/harness/fetch_submission.py \
        --submission-id 180909 --out champion-code/round-66/top-raw
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def _find_apex_root() -> Path:
    path = Path.cwd().resolve()
    for _ in range(10):
        if (path / ".apex.config.json").exists():
            return path
        if path.parent == path:
            break
        path = path.parent
    return Path.cwd()


async def _fetch(submission_id: int, out: Path, hotkey_file: str, timeout: float) -> None:
    from cli.utils.client import Client
    from common.models.api.code import CodeRequest
    from common.models.api.submission import FileRequest, SubmissionRequest, SubmissionResponse

    async with Client(hotkey_file_path=hotkey_file, timeout=timeout) as client:
        resp = await client._make_request(method="GET", path="/miner/submission",
                                          params=SubmissionRequest(submission_id=submission_id)
                                          .model_dump())
        found = SubmissionResponse.model_validate(resp.json()).submissions
        if not found:
            raise SystemExit(f"submission {submission_id} not found")
        submission = found[0]
        detail = await client.get_submission_detail(submission_id)
        out.mkdir(parents=True, exist_ok=True)

        if detail is not None and detail.eval_metadata is not None:
            (out / "metadata.json").write_text(
                json.dumps(detail.eval_metadata.model_dump(mode="json"), indent=2))
            print("metadata.json")

        code_name = os.path.basename(detail.code_path) if detail and detail.code_path \
            else f"submission_{submission_id}.py"
        parts, start = [], 0
        while True:
            code = await client.get_submission_code(CodeRequest(
                competition_id=submission.competition_id,
                round_number=submission.round_number,
                hotkey=submission.hotkey,
                version=submission.version,
                start_idx=start))
            if code is None or not code.code:
                break
            parts.append(code.code)
            nxt = code.pagination.next_start_idx if code.pagination else None
            if nxt is None or nxt <= start:
                break
            start = nxt
        if parts:
            (out / code_name).write_text("".join(parts))
            print(f"{code_name}: {sum(len(p) for p in parts)} characters")
        else:
            print("code: not available (not yet revealed?)")

        paths = (detail.eval_file_paths or {}) if detail else {}
        for file_type, names in paths.items():
            for path in (names if isinstance(names, list) else [names]):
                name = os.path.basename(path) if isinstance(path, str) else str(path)
                chunks, start = [], 0
                while True:
                    data = await client.get_file_chunked(FileRequest(
                        submission_id=submission_id, file_type=file_type.lower(),
                        file_name=name, start_idx=start, reverse=False))
                    if data is None or not data.data:
                        break
                    chunks.append(data.data)
                    nxt = data.pagination.next_start_idx
                    if nxt is None or nxt <= start:
                        break
                    start = nxt
                if chunks:
                    (out / name).write_text("".join(chunks))
                    print(f"{name}: {sum(len(c) for c in chunks)} characters")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--submission-id", "-s", type=int, required=True)
    parser.add_argument("--out", "-o", type=Path, required=True)
    args = parser.parse_args()

    try:
        from cli.utils.config import Config
    except ImportError:
        print("Run from an Apex checkout: uv run --group dev python .../fetch_submission.py",
              file=sys.stderr)
        raise SystemExit(1)
    config = Config.load_config(config_file_path=_find_apex_root() / ".apex.config.json")
    if not config.hotkey_file_path:
        raise SystemExit("No hotkey linked. Run `apex link` first.")
    asyncio.run(_fetch(args.submission_id, args.out, config.hotkey_file_path, config.timeout))


if __name__ == "__main__":
    main()
