"""Build and encode a replica pool from Subnet 13 (Gravity) data.

The platform's social subsets are X and Reddit posts crawled by Subnet 13
miners, English only, at least ~80 characters. SN13 miners publish the same
crawls on Hugging Face (`*/reddit_dataset_*`, `*/x_dataset_*`), so this pool is
drawn from those shards rather than from unrelated public corpora.

Communities stand in for crawl topics: a subreddit for Reddit, a keyword
family for X (Gravity's X tasks are keyword searches, and most SN13 X rows
carry no hashtag label). Each community contributes a bounded number of
texts, posts first, so a subset can be assembled from a handful of them the
way a round is assembled from a handful of crawls.

    python3 harness/sn13_pool.py --shards /tmp/sn13/*.parquet --out /tmp/sn13
    python3 harness/sn13_pool.py --out /tmp/sn13 --encode

`--encode` embeds whatever `pool.jsonl` sits in `--out`, so an arXiv pool
(rows with a `text` field, e.g. recent titles from
`librarian-bots/arxiv-metadata-snapshot`) is encoded the same way.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

EN_WORDS = frozenset(
    "the be to of and a in that have i it for not on with he as you do at this "
    "but his by from they we say her she or an will my one all would there their "
    "what so up out if about who get which go me when make can like time no just "
    "him know take people into year your good some could them see other than then "
    "now look only come its over think also back after use two how our work first "
    "well way even new want because any these give day most us is are was were "
    "has had been".split())

X_TOPICS = {
    "x:crypto": r"\b(bitcoin|btc|crypto|ethereum|eth|solana|altcoin|blockchain|defi|memecoin)\b",
    "x:ai": r"\b(ai|chatgpt|openai|llm|gpt|artificial intelligence|machine learning|nvidia)\b",
    "x:politics": r"\b(trump|biden|election|congress|senate|democrats|republicans|gop|white house)\b",
    "x:football": r"\b(premier league|arsenal|chelsea|liverpool|man utd|manchester|goal|striker)\b",
    "x:usports": r"\b(nba|nfl|lakers|warriors|touchdown|quarterback|playoffs|super bowl)\b",
    "x:gaming": r"\b(gaming|playstation|xbox|nintendo|steam|gameplay|fortnite|gamer)\b",
    "x:music": r"\b(album|song|concert|tour|spotify|single|rapper|singer)\b",
    "x:film": r"\b(movie|film|netflix|trailer|episode|series|season finale|box office)\b",
    "x:markets": r"\b(stocks|stock market|nasdaq|s&p|inflation|fed|interest rates|recession)\b",
    "x:science": r"\b(nasa|space|climate|scientists|research|study finds|vaccine|spacex)\b",
}


def english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 40:
        return False
    if sum(c.isascii() for c in letters) / len(letters) < 0.95:
        return False
    words = re.findall(r"[a-z']+", text.lower())
    return len(words) >= 8 and sum(w in EN_WORDS for w in words) / len(words) >= 0.25


def load_rows(shards: list[str]) -> list[dict]:
    import pyarrow.parquet as pq

    rows, seen = [], set()
    for shard in shards:
        pf = pq.ParquetFile(shard)
        names = set(pf.schema_arrow.names)
        reddit = "communityName" in names
        cols = ["text", "label"] + (["dataType", "communityName"] if reddit else [])
        for group in range(pf.num_row_groups):
            table = pf.read_row_group(group, columns=cols).to_pydict()
            for i, text in enumerate(table["text"]):
                if not text or len(text) < 80 or not english(text):
                    continue
                key = text.strip().lower()[:200]
                if key in seen:
                    continue
                seen.add(key)
                if reddit:
                    rows.append({"text": text, "src": "reddit",
                                 "kind": table["dataType"][i] or "comment",
                                 "community": (table["communityName"][i] or "").lower()})
                else:
                    rows.append({"text": text, "src": "x", "kind": "post",
                                 "community": (table["label"][i] or "null").lower()})
    return rows


def build(args) -> None:
    rows = load_rows(args.shards)
    rng = random.Random(args.seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["src"] != "reddit":
            continue
        if row["kind"] != "post" and len(row["text"]) < args.min_comment_chars:
            continue
        groups[row["community"]].append(row)
    compiled = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in X_TOPICS.items()}
    for row in rows:
        if row["src"] != "x":
            continue
        hits = [name for name, rx in compiled.items() if rx.search(row["text"])]
        if len(hits) == 1:
            groups[hits[0]].append(dict(row, community=hits[0]))

    reddit_names = sorted(n for n, g in groups.items()
                          if not n.startswith("x:") and len(g) >= args.min_community)
    rng.shuffle(reddit_names)
    chosen = reddit_names[:args.reddit_communities]
    chosen += sorted(n for n in groups if n.startswith("x:")
                     and len(groups[n]) >= args.per_community // 2)

    pool = []
    for name in chosen:
        members = groups[name][:]
        rng.shuffle(members)
        if name.startswith("x:"):
            pool.extend(members[:args.per_community])
            continue
        posts = [m for m in members if m["kind"] == "post"][:args.max_posts]
        rest = [m for m in members if m["kind"] != "post"]
        pool.extend((posts + rest)[:args.per_community])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pool.jsonl", "w") as fh:
        for row in pool:
            fh.write(json.dumps(row) + "\n")
    kinds = defaultdict(int)
    for row in pool:
        kinds[(row["src"], row["kind"])] += 1
    print(f"{len(pool)} texts from {len(chosen)} communities: {dict(kinds)}")


def encode(args) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    pool = [json.loads(line) for line in open(out / "pool.jsonl")]
    texts = [row["text"] for row in pool]
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    order = np.argsort([len(t) for t in texts])
    emb = np.zeros((len(texts), 768), np.float32)
    step = 2048
    for start in range(0, len(texts), step):
        idx = order[start:start + step]
        emb[idx] = model.encode([texts[i] for i in idx], batch_size=32,
                                normalize_embeddings=True)
        print(f"encoded {min(start + step, len(texts))}/{len(texts)}", flush=True)
        np.save(out / "pool_emb.partial.npy", emb)
    np.save(out / "pool_emb.npy", emb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="*", default=[])
    parser.add_argument("--out", default="/tmp/sn13")
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reddit-communities", type=int, default=80)
    parser.add_argument("--min-community", type=int, default=600)
    parser.add_argument("--per-community", type=int, default=450)
    parser.add_argument("--max-posts", type=int, default=150)
    # SN13 comments run shorter than the platform's social texts (cleaned
    # median 180 against 230-330), so short comments are left out.
    parser.add_argument("--min-comment-chars", type=int, default=160)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.encode:
        encode(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
