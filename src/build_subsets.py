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


def build_combined_subset(scored_and_clustered_examples: list[dict[str, Any]], fraction: float) -> list[dict[str, Any]]:
    """Sample proportionally across clusters, prioritizing higher Q1+Q2 examples within each cluster.

    This is the full "Quality-Filtered Diverse Sampling" method: diversity
    (D1) sets how many examples come from each cluster, and quality (Q1+Q2)
    decides which examples within a cluster are chosen first.

    Args:
        scored_and_clustered_examples: Examples with "overlap_score", "q1_score",
            and "cluster_id" fields.
        fraction: Target subset size as a fraction of len(scored_and_clustered_examples).

    Returns:
        The combined quality+diversity subset of examples.
    """
    raise NotImplementedError("Implemented in Phase 5")


def main(config_path: str = None) -> None:
    config = load_config(config_path)
    seed = config["project"]["seed"]

    full = build_full_dataset_subset()
    logger.info("full.jsonl: %d examples", len(full))

    for fraction in config["experiment_matrix"]["subset_sizes"]:
        subset = build_random_subset(full, fraction, seed)
        logger.info("random_%dpct.jsonl: %d examples", int(fraction * 100), len(subset))


if __name__ == "__main__":
    main()
