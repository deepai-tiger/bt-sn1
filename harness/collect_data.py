"""Harvest a local corpus that stands in for the competition's round data.

Sources mirror the ones the platform draws from:
  * Reddit posts (subsets 1-3) -- webis/tldr-17
  * X posts (subsets 1-3)      -- enryu43/twitter100m_tweets
  * arXiv titles (subset 4)    -- gfissore/arxiv-abstracts-2021

All three arrive as parquet shards from the HF CDN. The datasets-server /rows
endpoint and arXiv's own API both rate-limit anonymous callers long before we
have enough volume, so whole-shard downloads are the only practical route.

Each record keeps a coarse `topic` label (subreddit, hashtag, or arXiv
category). The labels are never used as ground truth -- they exist so
`make_rounds.py` can assemble topically structured subsets the way Gravity's
keyword-targeted crawls do. Without that structure a random social sample is
semantically homogeneous and the ground-truth pipeline collapses it into one
blob, which is nothing like a real round.

    /tmp/venv/bin/python harness/collect_data.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

CORPUS_DIR = Path("/tmp/sn1_corpus")
SHARD_CACHE = CORPUS_DIR / "_shards"

# The platform gates on English, >=80 characters, and drops anchorless one-liners.
MIN_CHARS = 80
NON_ASCII_LIMIT = 0.25

_WS = re.compile(r"\s+")
_LATIN = re.compile(r"[A-Za-z]")
_HASHTAG = re.compile(r"#(\w{3,30})")
_URL = re.compile(r"https?://\S+")

# Hashtags that mark no topic at all, only engagement bait.
STOP_TAGS = {
    "rt", "follow", "followback", "like", "likeforlike", "retweet", "fyp", "viral",
    "love", "instagood", "photooftheday", "happy", "tbt", "repost", "giveaway",
    "news", "today", "new", "now", "day", "life", "the", "you", "and", "amp",
}


def _get(url: str, tries: int = 4, timeout: int = 900) -> bytes:
    delay = 4.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sn1-harness/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - transient network failures
            if attempt == tries - 1:
                raise
            print(f"  retry {attempt + 1}/{tries} ({exc})", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def looks_english(text: str, min_chars: int = MIN_CHARS) -> bool:
    if len(text) < min_chars:
        return False
    stripped = text.replace(" ", "")
    if not stripped:
        return False
    if len(_LATIN.findall(text)) < 0.5 * len(stripped):
        return False
    return sum(1 for ch in text if ord(ch) > 127) <= NON_ASCII_LIMIT * len(text)


def shard_urls(dataset: str, config: str = "default", split: str = "train") -> list[str]:
    index = json.loads(_get(f"https://huggingface.co/api/datasets/{dataset}/parquet"))
    return index[config][split]


def local_shard(dataset: str, url: str) -> Path:
    SHARD_CACHE.mkdir(parents=True, exist_ok=True)
    path = SHARD_CACHE / (dataset.replace("/", "_") + "_" + url.rstrip("/").split("/")[-1])
    if not path.exists():
        print(f"  downloading {url}", file=sys.stderr)
        path.write_bytes(_get(url))
    return path


def iter_shard_rows(dataset: str, columns: list[str], shards: int, seed: int):
    import pyarrow.parquet as pq

    urls = shard_urls(dataset)
    picked = random.Random(seed).sample(urls, k=min(shards, len(urls)))
    for url in picked:
        reader = pq.ParquetFile(local_shard(dataset, url))
        for batch in reader.iter_batches(batch_size=20000, columns=columns):
            cols = [batch.column(i).to_pylist() for i in range(len(columns))]
            yield from zip(*cols)


def collect_reddit(target: int, seed: int, shards: int) -> list[dict]:
    out, seen = [], set()
    for content, subreddit in iter_shard_rows(
            "webis/tldr-17", ["content", "subreddit"], shards, seed):
        text = _WS.sub(" ", str(content or "")).strip()
        if not looks_english(text) or len(text) > 1400:
            continue
        key = text[:160].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": text, "topic": f"r/{subreddit}", "source": "reddit"})
        if len(out) >= target:
            break
    return out


def collect_tweets(target: int, seed: int, shards: int) -> list[dict]:
    """Keep only hashtagged tweets; the hashtag is the topical anchor."""
    out, seen = [], set()
    for (tweet,) in iter_shard_rows(
            "enryu43/twitter100m_tweets", ["tweet"], shards, seed):
        text = _WS.sub(" ", str(tweet or "")).strip()
        if not looks_english(text) or len(text) > 400:
            continue
        tags = [t.lower() for t in _HASHTAG.findall(text)]
        tags = [t for t in tags if t not in STOP_TAGS and not t.isdigit()]
        if not tags:
            continue
        key = _URL.sub("", text)[:140].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": text, "topic": f"#{tags[0]}", "source": "x"})
        if len(out) >= target:
            break
    return out


def collect_arxiv(target: int, seed: int, shards: int) -> list[dict]:
    out, seen = [], set()
    for title, categories in iter_shard_rows(
            "gfissore/arxiv-abstracts-2021", ["title", "categories"], shards, seed):
        text = _WS.sub(" ", str(title or "")).strip()
        if not 25 <= len(text) <= 400:
            continue
        if sum(1 for ch in text if ord(ch) > 127) > 0.15 * len(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cats = categories if isinstance(categories, str) else " ".join(categories or [])
        out.append({"text": text, "topic": (cats.split() or ["unknown"])[0],
                    "source": "arxiv"})
        if len(out) >= target:
            break
    return out


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    topics = len({r["topic"] for r in rows})
    print(f"wrote {len(rows)} rows across {topics} topics -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reddit", type=int, default=120000)
    parser.add_argument("--tweets", type=int, default=120000)
    parser.add_argument("--arxiv", type=int, default=60000)
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", choices=["reddit", "tweets", "arxiv"], default=None)
    args = parser.parse_args()

    jobs = {
        "reddit": (CORPUS_DIR / "reddit.jsonl",
                   lambda: collect_reddit(args.reddit, args.seed + 1, args.shards)),
        "tweets": (CORPUS_DIR / "tweets.jsonl",
                   lambda: collect_tweets(args.tweets, args.seed + 2, args.shards)),
        "arxiv": (CORPUS_DIR / "arxiv.jsonl",
                  lambda: collect_arxiv(args.arxiv, args.seed + 3, 3)),
    }

    for name, (path, fn) in jobs.items():
        if args.only and args.only != name:
            continue
        if path.exists() and not args.force:
            print(f"{name}: {sum(1 for _ in path.open())} rows already present, skipping")
            continue
        print(f"{name}: collecting ...", flush=True)
        write(path, fn())


if __name__ == "__main__":
    main()
