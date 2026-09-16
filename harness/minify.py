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
        # Annotations are load-bearing here, in two different ways, and
        # `remove_annotations=True` breaks both.
        #
        # Class attribute annotations: it rewrites `texts: list[str]` to
        # `texts: 0`, which is valid Python that pydantic rejects at
        # class-creation time, so the submission fails on import.
        #
        # Argument annotations: FastAPI reads `request: ClusterRequest` to
        # decide that the parameter is the request *body*. Strip it and the
        # parameter becomes a query parameter, so every POST /cluster comes
        # back 422 and every subset scores zero -- with a process that starts
        # cleanly and answers /health, which is what makes it so quiet.
        remove_annotations=RemoveAnnotationsOptions(
            remove_variable_annotations=True,
            remove_return_annotations=True,
            remove_argument_annotations=False,
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
    """Import the build, then serve it and score it over real HTTP.

    Renaming is only *supposed* to be semantics-preserving, and twice now it
    silently was not, both times in the FastAPI layer rather than in anything
    the clustering tests touch. Importing the module and calling `make_app()`
    caught the first one and sailed straight past the second, because a
    missing argument annotation produces an app that builds perfectly and
    then 422s every request.

    So the check has to be the platform's own: start the file as a subprocess,
    wait for /health, POST to /cluster, and look at what comes back. Anything
    less does not exercise the part that has actually been breaking.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("submission_smoke_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_smoke_test"] = module
    spec.loader.exec_module(module)

    if module.make_app() is None:
        raise SystemExit("make_app() returned nothing")
    labels = module.cluster_texts([f"a short document about topic {i % 7}"
                                   for i in range(120)])
    if len(labels) != 120:
        raise SystemExit(f"cluster_texts returned {len(labels)} labels for 120 texts")
    if module.DISTILLED_BLOB and module.load_distilled() is None:
        raise SystemExit("the baked blob does not decode in the minified build")

    served = serve_and_call(path)
    print(f"smoke test: imports, clusters, decodes the blob, and answers "
          f"POST /cluster over HTTP ({served} ids)")


def serve_and_call(path: Path, texts: int = 400) -> int:
    """Run the build the way the platform does and POST one batch to it."""
    import json
    import socket
    import subprocess
    import time
    import urllib.error
    import urllib.request

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = subprocess.Popen([sys.executable, str(path), "--port", str(port)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True)
    try:
        deadline = time.time() + 60
        while True:
            if server.poll() is not None:
                raise SystemExit(f"server exited {server.returncode}:\n"
                                 f"{server.stdout.read()[-2000:]}")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=2) as reply:
                    if json.loads(reply.read())["status"] == "healthy":
                        break
            except (urllib.error.URLError, OSError, KeyError, ValueError):
                if time.time() > deadline:
                    raise SystemExit("server never reported healthy")
                time.sleep(0.3)

        body = json.dumps({"texts": [f"a short document about topic {i % 9}"
                                     for i in range(texts)]}).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/cluster", body,
            {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as reply:
                payload = json.loads(reply.read())
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"POST /cluster returned {exc.code}: "
                             f"{exc.read().decode()[:400]}")

        ids = payload.get("cluster_ids")
        if not isinstance(ids, list) or len(ids) != texts:
            raise SystemExit(f"POST /cluster returned {len(ids or [])} ids "
                             f"for {texts} texts")
        if not all(isinstance(i, int) for i in ids):
            raise SystemExit("cluster_ids are not all integers")
        return len(ids)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
