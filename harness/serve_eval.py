"""Score a submission the way the platform does: over HTTP, as a subprocess.

`evaluate.py` imports `cluster_texts` and calls it directly. That is fast and
it is what you want while tuning, but it skips the entire serving layer -- and
the serving layer is what has broken in practice. A submission scored 0.0 on
the platform while `evaluate.py` gave it 0.3801, because the minifier had
dropped the endpoint's argument annotation: FastAPI then treated the request
body as a query parameter and answered every POST with 422. The process
started cleanly and reported healthy the whole time.

This runs the file the way the container does -- `python <file> --port N` --
waits for /health, POSTs each round to /cluster, and scores the replies. Run
it on `build/submission.min.py` before every submission. It is the only check
that covers request parsing, response validation and JSON serialization.

    /tmp/venv/bin/python harness/serve_eval.py build/submission.min.py
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import score_subset  # noqa: E402

ROUNDS_DIR = Path("/tmp/sn1_rounds")
MEMORY_LIMIT_MIB = 1536


class PeakMemory(threading.Thread):
    """Track the served process's high-water mark against the container cap.

    Exceeding it gets the container killed mid-request, which from the
    outside looks the same as any other way of scoring zero. Worth watching
    here because the co-occurrence matrix scales with the batch vocabulary,
    not with anything the harness controls.
    """

    def __init__(self, pid: int) -> None:
        super().__init__(daemon=True)
        self.status = Path(f"/proc/{pid}/status")
        self.peak_mib = 0.0
        self.stop = False

    def run(self) -> None:
        while not self.stop:
            try:
                for line in self.status.read_text().splitlines():
                    if line.startswith("VmHWM"):
                        self.peak_mib = max(self.peak_mib,
                                            int(line.split()[1]) / 1024)
                        break
            except OSError:
                return
            time.sleep(0.05)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def wait_for_health(port: int, server: subprocess.Popen, timeout: float) -> float:
    """Block until /health answers, returning how long startup took."""
    started = time.time()
    deadline = started + timeout
    while True:
        if server.poll() is not None:
            output = server.stdout.read() if server.stdout else ""
            raise SystemExit(f"server exited {server.returncode} before becoming "
                             f"healthy:\n{output[-4000:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                        timeout=2) as reply:
                if json.loads(reply.read()).get("status") == "healthy":
                    return time.time() - started
        except (urllib.error.URLError, OSError, ValueError):
            pass
        if time.time() > deadline:
            raise SystemExit(f"/health never answered within {timeout:.0f}s")
        time.sleep(0.25)


def post_cluster(port: int, texts: list[str], timeout: float) -> tuple[list[int], float]:
    body = json.dumps({"texts": texts}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/cluster", body,
                                     {"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            payload = json.loads(reply.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:600]
        raise SystemExit(f"POST /cluster returned HTTP {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"POST /cluster failed: {exc}")
    elapsed = time.time() - started

    ids = payload.get("cluster_ids")
    if not isinstance(ids, list):
        raise SystemExit(f"response has no cluster_ids list: {str(payload)[:300]}")
    if len(ids) != len(texts):
        raise SystemExit(f"got {len(ids)} ids for {len(texts)} texts")
    return ids, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--rounds-dir", type=Path, default=ROUNDS_DIR)
    parser.add_argument("--subset", action="append", default=None)
    parser.add_argument("--budget", type=float, default=90.0)
    args = parser.parse_args()

    files = sorted(args.rounds_dir.glob("round_*_subset_*.json"))
    if args.subset:
        files = [f for f in files if any(s in f.stem for s in args.subset)]
    if not files:
        raise SystemExit(f"no rounds in {args.rounds_dir}")

    port = free_port()
    server = subprocess.Popen([sys.executable, str(args.submission),
                               "--port", str(port)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True)
    scores: list[float] = []
    memory = PeakMemory(server.pid)
    memory.start()
    try:
        startup = wait_for_health(port, server, timeout=60)
        print(f"{args.submission} healthy after {startup:.2f}s on port {port}\n")
        header = (f"{'round':<30}{'ARI':>8}{'NMI':>9}{'combined':>10}"
                  f"{'secs':>7}{'k':>7}")
        print(header)
        print("-" * len(header))
        for path in files:
            payload = json.loads(path.read_text())
            ids, elapsed = post_cluster(port, list(payload["texts"]), args.budget)
            result = score_subset(np.asarray(payload["labels"], np.int64),
                                  np.asarray(ids, np.int64))
            scores.append(result["combined"])
            flag = "  OVER BUDGET" if elapsed > args.budget else ""
            print(f"{payload['name']:<30}{result['ari']:>8.4f}{result['nmi']:>9.4f}"
                  f"{result['combined']:>10.4f}{elapsed:>7.1f}"
                  f"{len(set(ids)):>7}{flag}")
        print("-" * len(header))
        print(f"MEAN combined: {float(np.mean(scores)):.4f}   (n={len(scores)})")
        print(f"peak memory:   {memory.peak_mib:.0f} MiB of "
              f"{MEMORY_LIMIT_MIB} MiB")
        if memory.peak_mib > MEMORY_LIMIT_MIB:
            raise SystemExit("OVER THE MEMORY LIMIT")
    finally:
        memory.stop = True
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
