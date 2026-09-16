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

# ground-truth env (needs torch, so keep it separate)
uv venv /tmp/gtvenv --python 3.12
uv pip install --python /tmp/gtvenv/bin/python \
    sentence-transformers umap-learn scikit-learn "numpy<2.4" pyarrow
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
/tmp/venv/bin/python harness/grid.py --spec harness/specs/titles_c.json --subset arxiv
```

### Rebuilding the distilled embedding

The one baked artefact in the submission. Encoding is the slow part (~25
minutes on 4 CPU cores); the fit caches its normal equations, so tuning
`--dims`, `--alpha`, `--subspaces` and `--prune` afterwards costs ~25s a go.

```bash
/tmp/gtvenv/bin/python harness/distill_encode.py            # MiniLM targets
/tmp/venv/bin/python harness/distill_fit.py --dims 32 --subspaces 8 \
    --prune 0.5 --alpha 0.2 --word 6144 --char 2048         # fit and pack
/tmp/venv/bin/python harness/grid.py --spec harness/specs/distilled_w.json \
    --subset arxiv --blob /tmp/sn1_distill/blob.txt         # A/B before baking
/tmp/venv/bin/python harness/embed_blob.py /tmp/sn1_distill/blob.txt
/tmp/venv/bin/python harness/test_blob.py                   # format round trip
/tmp/venv/bin/python harness/minify.py                      # check the limit
```

## Files

| file | role |
|---|---|
| `collect_data.py` | pulls Reddit / X / arXiv parquet shards, keeps a coarse topic label per row |
| `make_rounds.py` | assembles topically structured subsets and bakes ground truth with the real pipeline |
| `score.py` | the competition metric: `(max(0, ARI) + NMI) / 2` |
| `evaluate.py` | imports a submission's `cluster_texts` and scores it across all rounds |
| `sweep.py` | prepared A/B experiments over `CONFIG` |
| `grid.py` | runs a JSON list of `CONFIG` overrides; can inject a candidate blob with `--blob` |
| `probe.py` | single-variant run, for quick one-offs |
| `specs/` | grid specs, one file per question asked |
| `distill_encode.py` | encodes the corpus with MiniLM to produce regression targets |
| `distill_fit.py` | solves and packs the distilled embedding |
| `embed_blob.py` | writes a trained blob into `DISTILLED_BLOB` |
| `test_blob.py` | round-trips the packer against the submission's unpacker |
| `minify.py` | builds the submission-sized copy and checks the character limit |
| `reference_champion.py` | readable reconstruction of the 4th-place submission, used as the A/B baseline |

## Where the ceiling is

`diagnose.py` answers the question that decides what is worth working on. It
assigns every point to whichever *ground-truth* cluster has the nearest
centroid in the submission's own feature space -- the best any clusterer could
do with these features if it were handed the true cluster locations for free:

| | pipeline | oracle | kmeans(true k) | ward(true k) |
|---|---|---|---|---|
| social subset | **0.418** | 0.417 | 0.376 | 0.392 |
| arXiv subset | **0.288** | 0.260 | 0.248 | 0.243 |

The pipeline already matches the oracle on social text and beats it on titles.
It can do that because the metric rewards output *shape* -- the noise bucket,
the granularity -- and not only correct assignment, which the oracle gets right
by construction.

The consequence is worth being explicit about: **clustering is not the
bottleneck, the feature space is.** This is consistent with the hyperparameter
searches, where all 13 social variants land within +/-0.005 of the default.
Cluster ensembles, better cut selection and alternative linkages all have a
ceiling of roughly where the submission already sits. The only lever with room
left is representation quality, which means a higher-fidelity distilled
embedding (currently reproducing 53% of MiniLM's pairwise geometry).

## What the harness cannot reproduce

**The platform's own distilled embedding.** The champion's two `lzma` blobs
(`Ak`, `BD`) were blanked to `'AAA'` and `''` in the CLI export, so roughly 31k
characters of quantized projection-matrix weights are gone.
`reference_champion.py` runs without them, which is why it scores 0.281 here
against the platform's reported 0.405 -- the A/B against it is fair, but it is
not a like-for-like reconstruction of the real submission's score. The current
submission ships its own blob, retrained from scratch (see `distill_fit.py`).

**The exact embedding model.** The competition says "sentence-transformer"
without naming one. This assumes `all-MiniLM-L6-v2`, by far the most common
default. If the platform uses something else, the distilled embedding is
distilled from the wrong teacher -- though sentence-embedding spaces correlate
strongly enough that it should still carry most of its value.

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
