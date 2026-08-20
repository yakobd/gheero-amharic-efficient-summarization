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

Scaffolding only for Phase 3 — implemented in Phase 4/5.
"""

from pathlib import Path
from typing import Any


def build_random_subset(examples: list[dict[str, Any]], fraction: float, seed: int) -> list[dict[str, Any]]:
    """Uniformly sample a fraction of examples at random (the random_* baseline conditions).

    Args:
        examples: Full pool of {id, article, summary} dicts to sample from.
        fraction: Target subset size as a fraction of len(examples), e.g. 0.25.
        seed: Random seed, config["project"]["seed"].

    Returns:
        The sampled subset of examples.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


def build_quality_only_subset(scored_examples: list[dict[str, Any]], fraction: float) -> list[dict[str, Any]]:
    """Select the top-scoring fraction of examples by Q1+Q2 quality score only.

    Args:
        scored_examples: Examples with "overlap_score" and "q1_score" fields,
            from quality_scoring.score_dataset().
        fraction: Target subset size as a fraction of len(scored_examples).

    Returns:
        The top-quality subset of examples, with no diversity constraint.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


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
    raise NotImplementedError("Implemented in Phase 4/5")


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
    raise NotImplementedError("Implemented in Phase 4/5")


def build_all_subsets(config: dict, processed_dir: Path, subsets_dir: Path) -> None:
    """Build and save every subset in the 9-run experiment matrix.

    Iterates config["experiment_matrix"] (subset_sizes x conditions, plus the
    full_dataset ceiling run) and writes each resulting subset to
    data/subsets/<run_name>.jsonl.

    Args:
        config: Full project config, as returned by utils.load_config().
        processed_dir: Directory holding data/processed/train.jsonl (source pool).
        subsets_dir: Output directory, data/subsets/.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


if __name__ == "__main__":
    raise NotImplementedError("Implemented in Phase 4/5")
