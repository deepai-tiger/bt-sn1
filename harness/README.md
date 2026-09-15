# Offline harness for SN1 / Apex text clustering

Every tuned constant in a competitive submission was found by searching against
a score. The platform gives you 4 submissions per 24 hours, so that search has
to happen locally. This harness reproduces the scoring loop end to end.

## Why a local replica is possible

The competition states the ground truth comes from
`sentence-transformer -> UMAP -> HDBSCAN`, and `champion-code/metadata.json`
leaks the one parameter that matters: every subset reports a minimum
ground-truth cluster size of 25-27, which pins HDBSCAN's `min_cluster_size` at
25. Reproducing the rest with library defaults lands within a few clusters of
the platform's reported shape.

| | local replica | platform (round 0052) |
|---|---|---|
| social clusters | 44-54 | 34-44 |
| social noise | 26-32% | 13-20% |
| min cluster size | 25-27 | 25-27 |
| median cluster size | 50-70 | 61-75 |
| arXiv clusters | 29 | 41 |
| arXiv noise | 41% | 29% |

The replica is slightly harder than the real thing (more noise, so lower
absolute scores), which is the safe direction for a harness to err.

The strongest confirmation is behavioural: `reference_champion.py` emits
**1026 / 1026 / 1027** clusters on the three social subsets and **2058** on
arXiv. The platform recorded **1026 / 1026 / 1026** and **2020**. The
reconstruction is on the same code path with the same effective parameters.

## Setup

Two virtualenvs, because the submission must run against the sandbox's exact
package set while ground-truth generation needs a much heavier stack.

```bash
curl -sSL https://astral.sh/uv/install.sh | sh

# submission env -- mirrors base-code/dockerfiles/requirements.txt
uv venv /tmp/venv --python 3.12
uv pip install --python /tmp/venv/bin/python \
    numpy==2.3.5 scikit-learn==1.6.1 fastapi==0.124.2 uvicorn==0.38.0 \
    pydantic==2.12.5 pyarrow

# ground-truth env
uv venv /tmp/gtvenv --python 3.12
uv pip install --python /tmp/gtvenv/bin/python \
    sentence-transformers umap-learn scikit-learn "numpy<2.4"
```

## Usage

```bash
# 1. Harvest ~460k real texts (Reddit, X, arXiv) into /tmp/sn1_corpus
/tmp/venv/bin/python harness/collect_data.py

# 2. Bake ground truth for 3 rounds x 4 subsets into /tmp/sn1_rounds
/tmp/gtvenv/bin/python harness/make_rounds.py --rounds 3

# 3. Score a submission
/tmp/venv/bin/python harness/evaluate.py champion-code/code_submission_v1.py

# 4. Compare strategy variants
/tmp/venv/bin/python harness/sweep.py --experiment noise_mode
```

## Files

| file | role |
|---|---|
| `collect_data.py` | pulls Reddit / X / arXiv parquet shards, keeps a coarse topic label per row |
| `make_rounds.py` | assembles topically structured subsets and bakes ground truth with the real pipeline |
| `score.py` | the competition metric: `(max(0, ARI) + NMI) / 2` |
| `evaluate.py` | imports a submission's `cluster_texts` and scores it across all rounds |
| `sweep.py` | overrides `CONFIG` in the submission to A/B strategy choices |
| `reference_champion.py` | readable reconstruction of the 4th-place submission, used as the A/B baseline |

## Two things the harness cannot reproduce

**The distilled embedding.** The champion's two `lzma` blobs (`Ak`, `BD`) were
blanked to `'AAA'` and `''` in the CLI export, so roughly 31k characters of
quantized projection-matrix weights are gone. Both `reference_champion.py` and
the current submission run without them, which makes the A/B fair but means
local absolute scores sit well below the platform's 0.405.

**Sandbox speed.** This VM runs the same pipeline in roughly half the wall time
the platform recorded (10.7s here vs 21.8s there). Treat local timings as
needing a 2x safety factor; the submission's internal deadline checks use real
elapsed time, so they adapt on their own.

## Sampling caveat

`make_rounds.py` builds social subsets from topical slices (subreddits and
hashtags) because a uniform social sample is too homogeneous -- the real
pipeline collapses it into one 5000-point blob, nothing like a real round. Those
topic labels are a *sampling* device only, mirroring how Gravity's
keyword-targeted crawls compose a round. Ground truth always comes from the
embedding pipeline, never from the labels.
