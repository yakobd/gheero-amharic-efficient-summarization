"""Quality scoring for article-summary pairs: Q2 overlap filter + Q1 learnability score.

Q2 (overlap/compression filter): removes low-fidelity pairs where the summary
doesn't plausibly derive from the article (e.g. near-zero lexical overlap or an
implausible compression ratio), catching scraping/alignment errors in XL-Sum.

Q1 (learnability score): scores each surviving pair by the loss a PRETRAINED
(not fine-tuned) mT5-small assigns to the reference summary given the article.
Examples in the middle of the loss distribution are the ones the model can
plausibly learn from; examples that are near-zero loss are redundant/trivial
and examples with very high loss are likely noisy or misaligned.

Scaffolding only for Phase 3 — implemented in Phase 4/5.
"""

from typing import Any


def compute_overlap_score(article: str, summary: str) -> float:
    """Compute a lexical overlap/compression signal between an article and its summary.

    Used by the Q2 filter: pairs scoring below
    config["quality_scoring"]["min_overlap_score"] are dropped as likely
    misaligned or low-fidelity.

    Args:
        article: Full article text.
        summary: Reference summary text.

    Returns:
        A score in [0, 1] where higher indicates stronger evidence the
        summary was genuinely derived from the article.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


def apply_q2_filter(examples: list[dict[str, Any]], min_overlap_score: float) -> list[dict[str, Any]]:
    """Drop examples whose overlap score falls below min_overlap_score.

    Args:
        examples: List of {id, article, summary} dicts.
        min_overlap_score: Threshold from config["quality_scoring"]["min_overlap_score"].

    Returns:
        The subset of examples that pass the Q2 quality filter.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


def compute_reference_model_loss(examples: list[dict[str, Any]], reference_model_name: str) -> list[float]:
    """Compute per-example loss under a pretrained (not fine-tuned) reference model.

    Loads reference_model_name (config["quality_scoring"]["reference_model"])
    with no task-specific fine-tuning and computes the teacher-forced loss of
    generating each example's reference summary from its article. This loss
    is the raw signal behind the Q1 learnability score.

    Args:
        examples: List of {id, article, summary} dicts.
        reference_model_name: HF model name, e.g. "google/mt5-small".

    Returns:
        A list of per-example loss values, aligned with `examples`.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


def compute_q1_scores(
    losses: list[float],
    loss_lower_percentile: float,
    loss_upper_percentile: float,
) -> list[float]:
    """Convert raw reference-model losses into Q1 learnability scores.

    Examples whose loss falls within [loss_lower_percentile, loss_upper_percentile]
    of the loss distribution (config["quality_scoring"]) are treated as most
    learnable; scores taper off outside that band.

    Args:
        losses: Per-example reference-model losses.
        loss_lower_percentile: Lower percentile bound, e.g. 20.
        loss_upper_percentile: Upper percentile bound, e.g. 80.

    Returns:
        A list of Q1 scores, aligned with `losses`, higher meaning more learnable.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


def score_dataset(examples: list[dict[str, Any]], config: dict) -> list[dict[str, Any]]:
    """Run the full quality-scoring pipeline (Q2 filter then Q1 score) on a dataset.

    Args:
        examples: List of {id, article, summary} dicts, e.g. loaded from
            data/processed/train.jsonl.
        config: Full project config, as returned by utils.load_config().

    Returns:
        The Q2-filtered examples, each augmented with an "overlap_score" and
        a "q1_score" field, ready for use by diversity_clustering.py and
        build_subsets.py.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


if __name__ == "__main__":
    raise NotImplementedError("Implemented in Phase 4/5")
