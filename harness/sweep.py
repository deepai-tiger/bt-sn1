"""A/B strategy choices in the submission against the local rounds.

The submission keeps its knobs in a single `CONFIG` dict, so a variant is just
a set of overrides. Every experiment below answers one question that would
otherwise cost a real submission to test.

    /tmp/venv/bin/python harness/sweep.py --experiment noise_mode
    /tmp/venv/bin/python harness/sweep.py --list
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent))
from score import pred_stats, score_subset  # noqa: E402

ROUNDS_DIR = Path("/tmp/sn1_rounds")
SUBMISSION = Path(__file__).parent.parent / "champion-code" / "code_submission_v1.py"

# Each experiment maps a variant name to CONFIG overrides.
EXPERIMENTS: dict[str, dict[str, dict]] = {
    # Does ground truth's single -1 group reward one shared bucket, or is
    # abstaining with singletons better? This is the single highest-information
    # question about the metric.
    "noise_mode": {
        "bucket": {"noise_mode": "bucket"},
        "singleton": {"noise_mode": "singleton"},
        "none": {"noise_mode": "none"},
    },
    # Replicate the ground-truth clusterer, or cut a dendrogram?
    "cluster_mode": {
        "hdbscan": {"cluster_mode": "hdbscan"},
        "linkage_modularity": {"cluster_mode": "linkage", "k_mode": "modularity"},
        "linkage_fixed45": {"cluster_mode": "linkage", "k_mode": "fixed", "k_fixed": 45},
    },
    # Is a UMAP-like spectral space worth its cost?
    "reduce_mode": {
        "none": {"reduce_mode": "none"},
        "svd40": {"reduce_mode": "svd", "reduce_dim": 40},
        "svd80": {"reduce_mode": "svd", "reduce_dim": 80},
        "spectral20": {"reduce_mode": "spectral", "reduce_dim": 20},
        "spectral40": {"reduce_mode": "spectral", "reduce_dim": 40},
    },
    # Same question, but only the arXiv path, which measured the opposite way.
    "title_reduce_mode": {
        "spectral10": {"title_reduce_mode": "spectral", "title_reduce_dim": 10},
        "spectral20": {"title_reduce_mode": "spectral", "title_reduce_dim": 20},
        "none": {"title_reduce_mode": "none"},
        "svd40": {"title_reduce_mode": "svd", "title_reduce_dim": 40},
    },
    "title_noise_mode": {
        "bucket": {"title_noise_mode": "bucket"},
        "singleton": {"title_noise_mode": "singleton"},
        "none": {"title_noise_mode": "none"},
    },
    "title_min_cluster_size": {
        "15": {"title_min_cluster_size": 15},
        "25": {"title_min_cluster_size": 25},
        "35": {"title_min_cluster_size": 35},
    },
    # Ground truth has no cluster below 25 members.
    "hdbscan_min_samples": {
        "auto(=mcs)": {"hdbscan_min_samples": None},
        "8": {"hdbscan_min_samples": 8},
        "15": {"hdbscan_min_samples": 15},
        "40": {"hdbscan_min_samples": 40},
    },
    "hdbscan_selection": {
        "eom": {"hdbscan_selection": "eom"},
        "leaf": {"hdbscan_selection": "leaf"},
    },
    "noise_reclaim": {
        "0.0": {"noise_reclaim_thr": 0.0, "title_noise_reclaim_thr": 0.0},
        "0.5": {"noise_reclaim_thr": 0.5, "title_noise_reclaim_thr": 0.5},
        "0.7": {"noise_reclaim_thr": 0.7, "title_noise_reclaim_thr": 0.7},
        "0.85": {"noise_reclaim_thr": 0.85, "title_noise_reclaim_thr": 0.85},
    },
    "fuzzy_spectral": {
        "on": {"fuzzy_spectral": True},
        "off": {"fuzzy_spectral": False},
    },
    "fuzzy_smooth": {
        "social_off_titles_on": {"fuzzy_smooth": False, "title_fuzzy_smooth": True},
        "both_off": {"fuzzy_smooth": False, "title_fuzzy_smooth": False},
        "both_on": {"fuzzy_smooth": True, "title_fuzzy_smooth": True},
    },
    "small_cluster_mode": {
        "noise": {"small_cluster_mode": "noise"},
        "merge": {"small_cluster_mode": "merge"},
        "none": {"small_cluster_mode": "none"},
    },
    # How much of the batch should be abstained on beyond the clusterer's own
    # rejects? The previous submission used 0.20 on social and 0.40 on arXiv.
    "extra_noise": {
        "0.0": {"extra_noise_frac": 0.0},
        "0.1": {"extra_noise_frac": 0.1},
        "0.2": {"extra_noise_frac": 0.2},
    },
    "min_cluster_size": {
        "15": {"min_cluster_size": 15},
        "20": {"min_cluster_size": 20},
        "25": {"min_cluster_size": 25},
        "32": {"min_cluster_size": 32},
    },
    "smoothing": {
        "6iter": {"smooth_iters": 6},
        "8iter": {"smooth_iters": 8},
        "12iter": {"smooth_iters": 12},
        "18iter": {"smooth_iters": 18},
    },
    "smooth_alpha": {
        "0.15": {"smooth_alpha": 0.15},
        "0.25": {"smooth_alpha": 0.25},
        "0.4": {"smooth_alpha": 0.4},
        "0.55": {"smooth_alpha": 0.55},
    },
    "smooth_k": {
        "8": {"smooth_k": 8},
        "12": {"smooth_k": 12},
        "20": {"smooth_k": 20},
        "30": {"smooth_k": 30},
    },
    "features": {
        "balanced": {"w_lsa": 1.0, "w_ppmi": 1.2},
        "ppmi_only": {"w_lsa": 0.0, "w_ppmi": 1.4},
        "ppmi_heavy": {"w_lsa": 0.55, "w_ppmi": 1.4},
        "lsa_only": {"w_lsa": 1.0, "w_ppmi": 0.0},
    },
    "title_features": {
        "balanced": {"title_w_lsa": 1.0, "title_w_ppmi": 1.2},
        "lsa_heavy": {"title_w_lsa": 1.6, "title_w_ppmi": 0.9},
        "ppmi_heavy": {"title_w_lsa": 0.6, "title_w_ppmi": 1.5},
        "lsa_only": {"title_w_lsa": 1.0, "title_w_ppmi": 0.0},
    },
    "title_smoothing": {
        "2iter": {"title_smooth_iters": 2},
        "4iter": {"title_smooth_iters": 4},
        "8iter": {"title_smooth_iters": 8},
        "14iter": {"title_smooth_iters": 14},
    },
    "title_smooth_alpha": {
        "0.3": {"title_smooth_alpha": 0.3},
        "0.5": {"title_smooth_alpha": 0.5},
        "0.7": {"title_smooth_alpha": 0.7},
    },
    "title_smooth_k": {
        "10": {"title_smooth_k": 10},
        "20": {"title_smooth_k": 20},
        "35": {"title_smooth_k": 35},
    },
    "title_char_weight": {
        "0.45": {"title_char_weight": 0.45},
        "0.9": {"title_char_weight": 0.9},
        "1.5": {"title_char_weight": 1.5},
    },
    "title_drop_components": {
        "0": {"title_drop_components": 0},
        "1": {"title_drop_components": 1},
        "2": {"title_drop_components": 2},
    },
    "char_weight": {
        "0.0": {"char_weight": 0.0},
        "0.45": {"char_weight": 0.45},
        "0.9": {"char_weight": 0.9},
    },
    "drop_components": {
        "1": {"drop_components": 1},
        "2": {"drop_components": 2},
        "3": {"drop_components": 3},
        "4": {"drop_components": 4},
    },
    "spectral_knn": {
        "8": {"spectral_knn": 8},
        "15": {"spectral_knn": 15},
        "25": {"spectral_knn": 25},
    },
}


def load_submission():
    spec = importlib.util.spec_from_file_location("submission_under_test", SUBMISSION)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_under_test"] = module
    spec.loader.exec_module(module)
    return module


def load_rounds(subset_filter: list[str] | None) -> list[dict]:
    files = sorted(ROUNDS_DIR.glob("round_*_subset_*.json"))
    if subset_filter:
        files = [f for f in files if any(s in f.stem for s in subset_filter)]
    return [json.loads(f.read_text()) for f in files]


def run_variant(module, rounds: list[dict], overrides: dict) -> dict:
    base = copy.deepcopy(module.CONFIG)
    module.CONFIG.update(overrides)
    try:
        per_subset = []
        for payload in rounds:
            started = time.perf_counter()
            pred = module.cluster_texts(list(payload["texts"]))
            elapsed = time.perf_counter() - started
            per_subset.append({
                "subset": payload["name"],
                "seconds": elapsed,
                **score_subset(payload["labels"], pred),
                **pred_stats(pred),
            })
    finally:
        module.CONFIG.clear()
        module.CONFIG.update(base)

    social = [r for r in per_subset if "arxiv" not in r["subset"]]
    arxiv = [r for r in per_subset if "arxiv" in r["subset"]]
    return {
        "mean": sum(r["combined"] for r in per_subset) / len(per_subset),
        "social": sum(r["combined"] for r in social) / len(social) if social else None,
        "arxiv": sum(r["combined"] for r in arxiv) / len(arxiv) if arxiv else None,
        "max_seconds": max(r["seconds"] for r in per_subset),
        "mean_clusters": sum(r["num_clusters"] for r in per_subset) / len(per_subset),
        "per_subset": per_subset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--subset", action="append", default=None)
    parser.add_argument("--base", action="append", default=None,
                        metavar="KEY=VALUE",
                        help="override applied to every variant")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.list:
        for name, variants in EXPERIMENTS.items():
            print(f"{name}: {', '.join(variants)}")
        return

    module = load_submission()
    if args.base:
        for item in args.base:
            key, _, value = item.partition("=")
            module.CONFIG[key] = json.loads(value)
        print(f"base overrides: {args.base}")

    rounds = load_rounds(args.subset)
    if not rounds:
        raise SystemExit(f"no rounds in {ROUNDS_DIR}")
    print(f"{len(rounds)} subsets: {', '.join(r['name'] for r in rounds)}\n")

    names = args.experiment or list(EXPERIMENTS)
    everything = {}
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name!r}; try --list")
        print(f"=== {name} ===")
        print(f"{'variant':<22} {'mean':>7} {'social':>7} {'arxiv':>7} "
              f"{'secs':>6} {'clusters':>9}")
        outcomes = {}
        for variant, overrides in EXPERIMENTS[name].items():
            result = run_variant(module, rounds, overrides)
            outcomes[variant] = result
            social = f"{result['social']:.4f}" if result["social"] is not None else "-"
            arxiv = f"{result['arxiv']:.4f}" if result["arxiv"] is not None else "-"
            print(f"{variant:<22} {result['mean']:>7.4f} {social:>7} {arxiv:>7} "
                  f"{result['max_seconds']:>6.1f} {result['mean_clusters']:>9.0f}",
                  flush=True)
        best = max(outcomes, key=lambda v: outcomes[v]["mean"])
        print(f"-> best: {best} ({outcomes[best]['mean']:.4f})\n")
        everything[name] = outcomes

    if args.json_out:
        args.json_out.write_text(json.dumps(everything, indent=2))


if __name__ == "__main__":
    main()
