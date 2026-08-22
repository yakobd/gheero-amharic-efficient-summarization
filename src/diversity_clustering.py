"""Diversity clustering (D1): LaBSE embeddings + k-means for proportional sampling.

Embeds each article with sentence-transformers/LaBSE, clusters the embeddings
with k-means (config["diversity_clustering"]["n_clusters"]), and saves
per-example cluster membership so build_subsets.py can sample proportionally
across clusters -- prioritizing higher Q1+Q2 examples within each cluster --
to keep selected subsets topically diverse rather than concentrated in a few
clusters.
"""

import json
import logging
from collections import Counter
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from utils import PROJECT_ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_BATCH_SIZE = 32


def compute_labse_embeddings(texts: list[str], config: dict) -> np.ndarray:
    """Embed article texts with the LaBSE sentence-transformer.

    Args:
        texts: Article texts to embed.
        config: Full project config, as returned by utils.load_config().

    Returns:
        A (len(texts), embedding_dim) array of embeddings.
    """
    embedding_model_name = config["diversity_clustering"]["embedding_model"]
    model = SentenceTransformer(embedding_model_name)
    return model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def cluster_embeddings(embeddings: np.ndarray, config: dict) -> np.ndarray:
    """Run k-means over article embeddings to assign a cluster id to each example.

    Args:
        embeddings: (n_examples, embedding_dim) array from compute_labse_embeddings().
        config: Full project config, as returned by utils.load_config().

    Returns:
        A (n_examples,) array of cluster ids in [0, n_clusters).
    """
    dc_cfg = config["diversity_clustering"]
    kmeans = KMeans(n_clusters=dc_cfg["n_clusters"], random_state=dc_cfg["random_state"])
    return kmeans.fit_predict(embeddings)


def main(config_path: str = None) -> None:
    config = load_config(config_path)
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]

    with open(processed_dir / "train.jsonl", "r", encoding="utf-8") as f:
        dataset: list[dict[str, Any]] = [json.loads(line) for line in f]

    logger.info(
        "Embedding %d articles with %s...", len(dataset), config["diversity_clustering"]["embedding_model"]
    )
    embeddings = compute_labse_embeddings([example["article"] for example in dataset], config)

    logger.info("Clustering into %d clusters...", config["diversity_clustering"]["n_clusters"])
    labels = cluster_embeddings(embeddings, config)

    cluster_assignments = {example["id"]: int(label) for example, label in zip(dataset, labels)}

    out_path = processed_dir / "diversity_clusters.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cluster_assignments, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", out_path)

    cluster_sizes = Counter(cluster_assignments.values())
    for cluster_id in sorted(cluster_sizes):
        logger.info("Cluster %d: %d examples", cluster_id, cluster_sizes[cluster_id])


if __name__ == "__main__":
    main()
