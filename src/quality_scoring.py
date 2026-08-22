"""Quality scoring for article-summary pairs: Q1 learnability + Q2 fidelity.

Q1 (learnability): scores each pair by the loss a PRETRAINED (not
fine-tuned) mT5-small assigns to the reference summary given the article,
using the same "summarize: " + article -> summary format and tokenization
lengths (config["model"]) as train.py. Examples in the middle of the loss
distribution are the ones the model can plausibly learn from; near-zero
loss means redundant/trivial, very high loss means likely noisy/misaligned.

Q2 (fidelity/overlap): a word-level overlap ratio between summary and
article -- the fraction of the summary's unique words that also appear in
the article -- as a fast, interpretable proxy for whether the summary is
actually grounded in the article's content, catching the known XL-Sum
issue where some summaries contain unsupported information.
"""

import json
import logging
import statistics
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils import PROJECT_ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUMMARIZE_PREFIX = "summarize: "
Q1_SCORING_BATCH_SIZE = 16


def compute_q1_learnability_scores(dataset: list[dict[str, Any]], config: dict) -> dict[str, float]:
    """Score each example by a pretrained reference model's teacher-forced loss.

    Loads config["quality_scoring"]["reference_model"] fresh (no
    fine-tuning, used only for scoring here) and computes the per-example
    loss of predicting each summary from its article, batched for speed.

    Args:
        dataset: List of {id, article, summary} dicts.
        config: Full project config, as returned by utils.load_config().

    Returns:
        Dict mapping example id -> loss value.
    """
    model_cfg = config["model"]
    reference_model_name = config["quality_scoring"]["reference_model"]

    tokenizer = AutoTokenizer.from_pretrained(reference_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(reference_model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    losses: dict[str, float] = {}

    with torch.no_grad():
        for start in range(0, len(dataset), Q1_SCORING_BATCH_SIZE):
            batch = dataset[start : start + Q1_SCORING_BATCH_SIZE]

            model_inputs = tokenizer(
                [SUMMARIZE_PREFIX + example["article"] for example in batch],
                max_length=model_cfg["max_input_length"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            label_ids = tokenizer(
                text_target=[example["summary"] for example in batch],
                max_length=model_cfg["max_target_length"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).input_ids.to(device)
            labels = label_ids.masked_fill(label_ids == tokenizer.pad_token_id, -100)

            outputs = model(**model_inputs, labels=labels)
            per_token_loss = loss_fct(
                outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1)
            ).view(labels.size())

            valid_mask = (labels != -100).float()
            per_example_loss = (per_token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

            for example, loss_value in zip(batch, per_example_loss.tolist()):
                losses[example["id"]] = loss_value

            logger.info(
                "Q1 scoring: %d/%d examples", min(start + Q1_SCORING_BATCH_SIZE, len(dataset)), len(dataset)
            )

    return losses


def compute_q2_fidelity_scores(dataset: list[dict[str, Any]]) -> dict[str, float]:
    """Compute a word-level overlap ratio between summary and article.

    Fraction of the summary's unique words that also appear in the
    article -- a fast, interpretable fidelity proxy for whether the
    summary is genuinely grounded in the article's content.

    Args:
        dataset: List of {id, article, summary} dicts.

    Returns:
        Dict mapping example id -> overlap score in [0, 1].
    """
    scores: dict[str, float] = {}
    for example in dataset:
        summary_words = set(example["summary"].split())
        article_words = set(example["article"].split())

        if not summary_words:
            scores[example["id"]] = 0.0
            continue

        scores[example["id"]] = len(summary_words & article_words) / len(summary_words)

    return scores


def _log_stats(name: str, values: list[float]) -> None:
    logger.info(
        "%s: min=%.4f max=%.4f mean=%.4f median=%.4f",
        name,
        min(values),
        max(values),
        statistics.mean(values),
        statistics.median(values),
    )


def main(config_path: str = None) -> None:
    config = load_config(config_path)
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]

    with open(processed_dir / "train.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    logger.info("Computing Q2 fidelity scores for %d examples...", len(dataset))
    q2_scores = compute_q2_fidelity_scores(dataset)

    logger.info(
        "Computing Q1 learnability scores for %d examples (reference model: %s)...",
        len(dataset),
        config["quality_scoring"]["reference_model"],
    )
    q1_scores = compute_q1_learnability_scores(dataset, config)

    combined = {
        example["id"]: {"q1_loss": q1_scores[example["id"]], "q2_overlap": q2_scores[example["id"]]}
        for example in dataset
    }

    out_path = processed_dir / "quality_scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", out_path)

    _log_stats("Q1 loss", list(q1_scores.values()))
    _log_stats("Q2 overlap", list(q2_scores.values()))


if __name__ == "__main__":
    main()
