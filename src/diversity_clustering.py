"""Diversity clustering (D1): LaBSE embeddings + k-means for proportional sampling.

Embeds each article with sentence-transformers/LaBSE, clusters the embeddings
with k-means (config["diversity_clustering"]["n_clusters"]), and prepares
per-cluster membership so build_subsets.py can sample proportionally across
clusters — prioritizing higher Q1+Q2 examples within each cluster — to keep
selected subsets topically diverse rather than concentrated in a few clusters.

Scaffolding only for Phase 3 — implemented in Phase 4/5.
"""

from typing import Any

import numpy as np


def compute_embeddings(texts: list[str], embedding_model_name: str) -> np.ndarray:
    """Embed a list of article texts with the LaBSE sentence-transformer.

    Args:
        texts: Article texts to embed.
        embedding_model_name: HF model name, e.g. "sentence-transformers/LaBSE"
            (config["diversity_clustering"]["embedding_model"]).

    Returns:
        A (len(texts), embedding_dim) array of embeddings.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    """Run k-means over article embeddings to assign a cluster id to each example.

    Args:
        embeddings: (n_examples, embedding_dim) array from compute_embeddings().
        n_clusters: Number of clusters, config["diversity_clustering"]["n_clusters"].
        random_state: Seed for reproducible clustering, config["diversity_clustering"]["random_state"].

    Returns:
        A (n_examples,) array of cluster ids in [0, n_clusters).
    """
    raise NotImplementedError("Implemented in Phase 4/5")


def assign_clusters(examples: list[dict[str, Any]], config: dict) -> list[dict[str, Any]]:
    """Embed and cluster a dataset, attaching a cluster id to each example.

    Args:
        examples: List of {id, article, summary, ...} dicts, typically already
            scored by quality_scoring.score_dataset().
        config: Full project config, as returned by utils.load_config().

    Returns:
        The input examples, each augmented with a "cluster_id" field, ready
        for proportional sampling in build_subsets.py.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


if __name__ == "__main__":
    raise NotImplementedError("Implemented in Phase 4/5")
