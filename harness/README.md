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

The one baked artefact in the submission. Encoding is the slow part (~12
minutes for 300k titles on 4 CPU cores); the fit caches its normal equations,
so tuning `--dims`, `--alpha`, `--subspaces` and `--prune` afterwards costs
~25s a go.

One map, trained on all three sources, serving both paths. It was briefly
specialized to arXiv titles, which was correct only while the social path had
the blob switched off; once reclaiming rejects made the blob useful on social
too, a shared map won (held out 0.4485 against 0.4309).

```bash
/tmp/gtvenv/bin/python harness/distill_encode.py            # MiniLM targets
/tmp/venv/bin/python harness/distill_fit.py --dims 32 --subspaces 8 \
    --prune 0.5 --alpha 0.1 --word 6144 --char 2048 \
    --reddit 90000 --tweets 110000 --arxiv 300000           # fit and pack
/tmp/venv/bin/python harness/grid.py --spec harness/specs/blob_choice.json \
    --blob /tmp/sn1_distill/blob.txt                        # A/B before baking
/tmp/venv/bin/python harness/embed_blob.py /tmp/sn1_distill/blob.txt
/tmp/venv/bin/python harness/test_blob.py                   # format round trip
/tmp/venv/bin/python harness/minify.py                      # check the limit
```

`distill_fit.py` drops any text that appears in a round under
`/tmp/sn1_rounds` or `/tmp/sn1_holdout`. Without that the overlap is total --
corpus and rounds are drawn from the same pools, so all 5000 texts of every
round are also training rows. It turns out not to matter much (an honest fit
scored 0.3236 against a leaky one's 0.3241: 8192x32 parameters cannot
memorize 300k texts), but the measurement is only trustworthy with it. The
filter lives at fit time rather than encode time so that re-cutting the rounds
does not cost another 12-minute encode.

Generating more rounds therefore shrinks the training pool, since rounds and
corpus compete for the same texts. At 36 subsets it costs 0.036 of fidelity
and nothing measurable in score (0.4349 against 0.4360), so prefer more
rounds.

Two knobs interact and should be moved together:

* **`--alpha` and `--prune`.** Pruning removes rows the fit is relying on, so
  the weaker the ridge the more pruning costs: at 50% pruning, `alpha=0.1`
  scores 0.627 and `alpha=0.01` only 0.542, even though `alpha=0.01` is the
  better *unpruned* fit. Mild pruning at `alpha=0.1` actually beats no
  pruning at all, acting as extra regularization.
* **`--prune-by`.** `norm` and `energy` are equivalent at the 50% we ship, but
  `energy` (row norm times the gram diagonal, so the bucket's real mass over
  the corpus) is far better when pruning hard: 0.53 versus 0.40 at 80%.

More `--dims` is nearly free at fixed `--subspaces` and still does not help:
the subvectors get wider, and quantization gives back more than the extra
dimensions earn (dims 64 at 8 subspaces reconstructs to 0.41, dims 32 to
0.50).

## Calibrating the replica, and why it matters more than anything else here

Running the platform's pipeline is not the same as reproducing its rounds, and
the difference is not academic: it cost a submission. The replica used to
assemble subsets from broad topical slices, which baked to

|  | clusters | noise |
|---|---|---|
| replica social | 41-54 | 26-39% |
| platform social | 34-49 | 13-22% |
| replica arXiv | 21-29 | 40-42% |
| platform arXiv | 39-41 | 26-29% |

A subreddit is not a topic, it is dozens of them, and a random sample over 154
arXiv categories is diffuse enough that HDBSCAN discards two fifths of it. So
the replica was scoring submissions against rounds a third of whose points
carried one shared noise label. Tuning on that rewarded lumping aggressively
and cutting coarsely, and the submission that won locally by 0.10 lost on the
platform by 0.10 -- an inverted harness, which is worse than no harness.

Slices are now the nearest neighbours of a seed post in embedding space, which
is what crawling a topic returns, with `spread` controlling how much of the
topic's fringe comes along. `--calibrate` sweeps the shape knobs and scores
them against the platform's reported statistics:

```bash
/tmp/gtvenv/bin/python harness/make_rounds.py --calibrate
```

Calibrated, the replica reproduces the platform's ordering for the first time:
the champion reconstruction scores 0.4010 against our then-0.3879.

Two cautions learned the hard way:

* **Calibrate against both statistics, not the mean of one.** The noise share
  drives structural decisions directly, and `metadata*.json` is the only
  window onto it. arXiv sitting 10 points higher in noise than social is not a
  detail; it changes which noise mode is optimal.
* **Do not chase the argmax cell.** A single seed draw swings the baked shape
  a long way -- n_topics 35 against 40 at the same tail and spread gave 22
  clusters at 2% noise against 41 at 15%. Pick central knobs and let
  individual subsets scatter, as the platform's own do.

## The noise share decides the output shape

`noise_sensitivity.py` scores all three noise modes on every subset and groups
by how much noise the ground truth actually holds. Over 36 subsets:

| ground-truth noise | n | reclaim | bucket | singleton |
|---|---|---|---|---|
| under 18% | 16 | **0.4990** | 0.4087 | 0.4773 |
| 18-24% | 8 | **0.4341** | 0.3718 | 0.4271 |
| over 24% | 12 | 0.3793 | 0.3612 | **0.3865** |

This is the single highest-leverage setting in the submission -- the spread
between best and worst mode is 0.09, several times any feature change -- and
it is entirely determined by a property of the data, not of the algorithm. It
is also why the miscalibrated replica was so damaging: at 30-42% noise the
table says bucket, and bucket is what the submission shipped.

Reclaiming wins below the crossover because a rejected point is usually a real
cluster member the clusterer was unsure of rather than something genuinely
unclusterable, and at 17% true noise the odds favour guessing a home for it.

The platform's social subsets sit at 13-22% and its arXiv at 26-29%, which
straddles the crossover, so the arXiv path is deliberately set to the
non-argmax choice: see the comment on `title_noise_mode`.

## Before submitting, run this

```bash
/tmp/venv/bin/python harness/minify.py                      # build + HTTP smoke test
/tmp/venv/bin/python harness/serve_eval.py build/submission.min.py
```

`evaluate.py` imports `cluster_texts` and calls it directly, which is what you
want while tuning but skips the entire serving layer. That gap cost a whole
round: a build scoring 0.3801 under `evaluate.py` scored **0.0** on the
platform, because the minifier had dropped the endpoint's argument annotation
and FastAPI fell back to reading the request body as a query parameter. Every
POST /cluster came back 422. Nothing looked wrong from outside -- the process
started, `/health` answered `healthy`, the reported evaluation error was
`None`, and the only hint was that scoring took 3.33s instead of ~75s.

`serve_eval.py` is the fix for that class of bug: it runs the file the way the
container does, waits for `/health`, POSTs each round to `/cluster`, scores
the replies, and tracks peak RSS against the 1536 MiB cap. It is the only
check here that covers request parsing, response validation, JSON
serialization and memory. Currently: 0.4349 tuning, 0.4452 held out (24
subsets), 614 MiB peak, matching the in-process numbers exactly. The champion
reconstruction scores 0.4010 and 0.4000 on the same rounds.

Two lessons worth keeping in mind, since both bugs came from the same place:

* **The minified build is the artefact, not the source.** Both failures were
  invisible in `code_submission_v1.py` and appeared only after minification.
  Never submit a build that has not been scored as a build.
* **Test the interface, not just the function.** The clustering is exercised
  constantly; the four lines of FastAPI wiring are where both bugs landed,
  because nothing was calling them.

## Files

| file | role |
|---|---|
| `collect_data.py` | pulls Reddit / X / arXiv parquet shards, keeps a coarse topic label per row |
| `make_rounds.py` | assembles focused subsets and bakes ground truth with the real pipeline; `--calibrate` fits its shape to the platform's |
| `calibrate_gt.py` | sweeps the baking side (HDBSCAN params) against the platform's reported stats |
| `noise_sensitivity.py` | scores every noise mode against each subset's true noise share |
| `score.py` | the competition metric: `(max(0, ARI) + NMI) / 2` |
| `evaluate.py` | imports a submission's `cluster_texts` and scores it across all rounds |
| `serve_eval.py` | serves a build as a subprocess and scores it over HTTP, as the platform does |
| `deframe_export.py` | recovers source from the CLI's Rich panel, flagging truncated lines |
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
| arXiv subset | **0.333** | 0.286 | 0.260 | 0.259 |

(Measured on the pre-calibration rounds; the conclusion held up, but the
absolute numbers are not comparable to anything else in this file.)

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
embedding (now reproducing 61% of MiniLM's pairwise geometry on titles, up
from 53%).

Even that lever is close to spent, and the reason is hash collisions. 300k
titles produce millions of distinct word n-grams competing for 6144 buckets,
so every bucket mixes hundreds of unrelated n-grams and the map can only ever
learn a coarse sketch. Widening the hash is the fix, and the character limit
forbids it: the bucket count sets the row count, which sets the size. Adding
training data no longer moves fidelity, which is what being collision-bound
rather than data-bound looks like.

## What the v2 champion export was worth

The v2 export (0.4117 on the platform, against v1's 0.4052) is framed by the
CLI the same way v1 was, so `deframe_export.py` recovers the logic but both of
its blobs are cut to ~100 characters -- Rich pads to a *display* width and CJK
codepoints are double-width, so the cut lands at ~100 characters rather than
the 200 that plain ASCII lines get. The weights stay unrecoverable.

The logic was still worth reading. v2 keeps **two** blobs, one per domain, and
routes titles through an otherwise separate pipeline. Testing each idea it
contains against this harness:

| v2 idea | result |
|---|---|
| separate per-domain blobs | **adopted**, as a titles-only fit: arXiv 0.2999 -> 0.3236 held out |
| a social blob at any weight | rejected: -0.002 held out, even trained on data overlapping the rounds |
| 80-way average-linkage cut on titles, 40% singletons | rejected: 0.266 against 0.320 |
| its smoothing (k=20, alpha=0.6, 4 iters) and char weight 0.35 | rejected: within noise on tuning, worse held out |
| whole-post co-occurrence instead of a 10-token window | rejected: 0.4127 against 0.4132 |
| merging clusters that share a non-Latin script | untestable: 5-40% script share to fire, these rounds run under 0.7% |
| sparse blob rows (delta-coded indices, 4-bit codes) | not needed: the budget is not the binding constraint |

The pattern in the rejections is that v2's shape choices are calibrated to a
blob far better than ours. Cutting to 80 clusters and discarding 40% of the
points as singletons pays off when the features can support that many
distinct, trustworthy groups; on ours it just fragments. The transferable part
was never a hyperparameter, it was the decision to specialize the map per
domain.

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
