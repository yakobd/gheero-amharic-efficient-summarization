"""Build the training subsets for all 9 runs in the experiment matrix.

Combines quality_scoring.py and diversity_clustering.py to materialize the
subset for each (subset_size, condition) pair defined in
config["experiment_matrix"]:

    - full_dataset:            100% of data/processed/train.jsonl, unfiltered
    - random_{25,50}pct:       uniform random sample, no quality/diversity signal
    - quality_only_{25,50}pct: top examples by Q1+Q2 score, no clustering
    - diversity_only_{25,50}pct: proportional per-cluster sampling, no quality signal
    - combined_{25,50}pct:     proportional per-cluster sampling, prioritizing
                                 higher Q1+Q2 examples within each cluster

Each subset is saved to data/subsets/<run_name>.jsonl for train.py to consume.

Only the baseline conditions (full_dataset, random_*) are implemented so far.
build_quality_only_subset, build_diversity_only_subset, and build_combined_subset
are scaffolding, pending Phase 5 once quality_scoring.py / diversity_clustering.py
are implemented.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from utils import PROJECT_ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_full_dataset_subset() -> list[dict[str, Any]]:
    """Copy data/processed/train.jsonl unchanged (the 100% ceiling condition).

    Returns:
        The full list of training examples, as written to data/subsets/full.jsonl.
    """
    config = load_config()
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    subsets_dir = PROJECT_ROOT / config["data"]["subsets_dir"]

    examples = _load_jsonl(processed_dir / "train.jsonl")
    _save_jsonl(examples, subsets_dir / "full.jsonl")
    return examples


def build_random_subset(dataset: list[dict[str, Any]], fraction: float, seed: int) -> list[dict[str, Any]]:
    """Uniformly sample a fraction of examples at random (the random_* baseline conditions).

    Args:
        dataset: Full pool of {id, article, summary} dicts to sample from.
        fraction: Target subset size as a fraction of len(dataset), e.g. 0.25.
        seed: Random seed, config["project"]["seed"].

    Returns:
        The sampled subset of examples, as written to
        data/subsets/random_<fraction*100>pct.jsonl.
    """
    config = load_config()
    subsets_dir = PROJECT_ROOT / config["data"]["subsets_dir"]

    rng = random.Random(seed)
    n = round(len(dataset) * fraction)
    subset = rng.sample(dataset, n)

    out_path = subsets_dir / f"random_{int(fraction * 100)}pct.jsonl"
    _save_jsonl(subset, out_path)
    return subset


def build_quality_only_subset(scored_examples: list[dict[str, Any]], fraction: float) -> list[dict[str, Any]]:
    """Select the top-scoring fraction of examples by Q1+Q2 quality score only.

    Args:
        scored_examples: Examples with "overlap_score" and "q1_score" fields,
            from quality_scoring.score_dataset().
        fraction: Target subset size as a fraction of len(scored_examples).

    Returns:
        The top-quality subset of examples, with no diversity constraint.
    """
    raise NotImplementedError("Implemented in Phase 5")


def build_diversity_only_subset(clustered_examples: list[dict[str, Any]], fraction: float, seed: int) -> list[dict[str, Any]]:
    """Sample proportionally across clusters, ignoring quality score.

    Args:
        clustered_examples: Examples with a "cluster_id" field, from
            diversity_clustering.assign_clusters().
        fraction: Target subset size as a fraction of len(clustered_examples).
        seed: Random seed, config["project"]["seed"].

    Returns:
        A subset proportionally covering every cluster, with no quality
        preference within clusters.
    """
    raise NotImplementedError("Implemented in Phase 5")


def build_combined_subset(
    dataset: list[dict[str, Any]],
    quality_scores: dict[str, dict[str, float]],
    cluster_assignments: dict[str, int],
    fraction: float,
    config: dict,
) -> list[dict[str, Any]]:
    """Q2-filter, then Q1-filter, then sample proportionally across clusters.

    This is the full "Quality-Filtered Diverse Sampling" method: first drop
    low-fidelity pairs (Q2), then keep only examples in the learnable band
    of the reference-model loss distribution (Q1, percentiles computed over
    the post-Q2 pool), then sample proportionally across clusters -- using
    the filtered pool's own cluster proportions -- prioritizing the
    lowest-q1_loss (most learnable) examples within each cluster when a
    cluster has more eligible examples than its proportional share.

    Args:
        dataset: Full pool of {id, article, summary} dicts (the original,
            unfiltered training set).
        quality_scores: {id: {"q1_loss", "q2_overlap"}}, from
            quality_scoring.main()'s data/processed/quality_scores.json.
        cluster_assignments: {id: cluster_label}, from
            diversity_clustering.main()'s data/processed/diversity_clusters.json.
        fraction: Target subset size as a fraction of len(dataset), e.g. 0.25.
        config: Full project config, as returned by utils.load_config().

    Returns:
        The combined quality+diversity subset of examples, as written to
        data/subsets/combined_<fraction*100>pct.jsonl.
    """
    qs_cfg = config["quality_scoring"]
    subsets_dir = PROJECT_ROOT / config["data"]["subsets_dir"]

    target_size = round(len(dataset) * fraction)

    q2_passed = [
        example for example in dataset if quality_scores[example["id"]]["q2_overlap"] >= qs_cfg["min_overlap_score"]
    ]
    n_dropped_q2 = len(dataset) - len(q2_passed)

    q2_losses = np.array([quality_scores[example["id"]]["q1_loss"] for example in q2_passed])
    lower_bound = np.percentile(q2_losses, qs_cfg["loss_lower_percentile"])
    upper_bound = np.percentile(q2_losses, qs_cfg["loss_upper_percentile"])

    filtered_pool = [
        example
        for example in q2_passed
        if lower_bound <= quality_scores[example["id"]]["q1_loss"] <= upper_bound
    ]
    n_dropped_q1 = len(q2_passed) - len(filtered_pool)

    by_cluster: dict[int, list[dict[str, Any]]] = {}
    for example in filtered_pool:
        by_cluster.setdefault(cluster_assignments[example["id"]], []).append(example)

    selected: list[dict[str, Any]] = []
    for members in by_cluster.values():
        cluster_quota = round(target_size * len(members) / len(filtered_pool))
        members_by_learnability = sorted(members, key=lambda ex: quality_scores[ex["id"]]["q1_loss"])
        selected.extend(members_by_learnability[:cluster_quota])

    out_path = subsets_dir / f"combined_{int(fraction * 100)}pct.jsonl"
    _save_jsonl(selected, out_path)

    logger.info(
        "combined_%dpct.jsonl: %d examples (target %d) -- excluded %d at Q2 (overlap < %.2f), "
        "excluded %d at Q1 (loss outside p%d-p%d band [%.4f, %.4f])",
        int(fraction * 100),
        len(selected),
        target_size,
        n_dropped_q2,
        qs_cfg["min_overlap_score"],
        n_dropped_q1,
        qs_cfg["loss_lower_percentile"],
        qs_cfg["loss_upper_percentile"],
        lower_bound,
        upper_bound,
    )

    return selected


def main(config_path: str = None) -> None:
    config = load_config(config_path)
    seed = config["project"]["seed"]
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]

    full = build_full_dataset_subset()
    logger.info("full.jsonl: %d examples", len(full))

    for fraction in config["experiment_matrix"]["subset_sizes"]:
        subset = build_random_subset(full, fraction, seed)
        logger.info("random_%dpct.jsonl: %d examples", int(fraction * 100), len(subset))

    with open(processed_dir / "quality_scores.json", "r", encoding="utf-8") as f:
        quality_scores = json.load(f)
    with open(processed_dir / "diversity_clusters.json", "r", encoding="utf-8") as f:
        cluster_assignments = json.load(f)

    # Scope cut to 25% combined only for now -- not combined_50pct,
    # quality_only, or diversity_only.
    build_combined_subset(full, quality_scores, cluster_assignments, fraction=0.25, config=config)


if __name__ == "__main__":
    main()
