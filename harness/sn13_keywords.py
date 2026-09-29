"""A keyword-crawl pool: what Gravity's X and Reddit tasks actually return.

`sn13_pool.py` groups texts by subreddit, and `make_rounds.py` slices them by
embedding neighbourhood. Neither is how the platform's data is gathered:
Gravity tasks are keyword and hashtag searches, so every post in a crawl
shares a literal term. That makes crawls lexically separable in a way
embedding slices are not, and it is a plausible reason the platform's rounds
bake with less noise (12-13% in round 66) than either replica does.

Each keyword below becomes a community of posts that contain it (word
boundary, case-insensitive), matching only one keyword so crawls do not
overlap. The output is a `pool.jsonl` that `sn13_pool.py --encode` embeds and
`sn13_rounds.py` samples, exactly like the subreddit pool.

    python3 harness/sn13_keywords.py --source /tmp/sn13/pool_all.jsonl --out /tmp/sn13_kw
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

KEYWORDS = [
    # AI and tech
    "chatgpt", "openai", "claude", "gemini", "nvidia", "deepseek", "llm", "iphone", "android",
    "tesla", "spacex", "windows", "linux", "gpu", "starlink", "apple", "google", "microsoft",
    # crypto and markets
    "bitcoin", "ethereum", "solana", "dogecoin", "crypto", "nft", "stocks", "inflation",
    "tariffs", "mortgage", "recession",
    # politics and world
    "trump", "biden", "musk", "doge", "ukraine", "russia", "israel", "gaza", "china", "iran",
    "brexit", "canada", "immigration", "election", "congress", "supreme court", "abortion",
    # sports
    "nba", "nfl", "premier league", "arsenal", "liverpool", "messi", "ronaldo", "lebron",
    "cricket", "f1", "ufc", "wrestling", "golf", "tennis", "olympics", "super bowl",
    # games
    "minecraft", "fortnite", "pokemon", "zelda", "elden ring", "gta", "call of duty",
    "nintendo", "playstation", "xbox", "steam", "valorant", "league of legends", "genshin",
    # entertainment
    "taylor swift", "netflix", "marvel", "star wars", "anime", "kpop", "spotify", "oscars",
    "disney", "harry potter", "game of thrones", "the office", "severance",
    # life and science
    "covid", "vaccine", "cancer", "diet", "gym", "climate", "nasa", "mars", "dog", "cat",
    "wedding", "pregnant", "college", "salary", "landlord", "therapy", "adhd", "autism",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/tmp/sn13/pool_all.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/sn13_kw"))
    parser.add_argument("--per-keyword", type=int, default=350)
    parser.add_argument("--min-keyword", type=int, default=200)
    parser.add_argument("--min-comment-chars", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    patterns = [(k, re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE)) for k in KEYWORDS]
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in open(args.source):
        row = json.loads(line)
        text = row["text"]
        if row["src"] == "reddit" and row["kind"] != "post" and len(text) < args.min_comment_chars:
            continue
        hits = [k for k, rx in patterns if rx.search(text)]
        if len(hits) == 1:
            groups[hits[0]].append({"text": text, "src": row["src"], "kind": row["kind"],
                                    "community": "kw:" + hits[0]})
    rng = random.Random(args.seed)
    pool = []
    for key in sorted(groups):
        members = groups[key]
        if len(members) < args.min_keyword:
            continue
        rng.shuffle(members)
        pool.extend(members[:args.per_keyword])
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "pool.jsonl", "w") as fh:
        for row in pool:
            fh.write(json.dumps(row) + "\n")
    kept = sorted({row["community"] for row in pool})
    print(f"{len(pool)} texts from {len(kept)} keywords")
    print({k: len(groups[k]) for k in sorted(groups, key=lambda k: -len(groups[k]))})


if __name__ == "__main__":
    main()
