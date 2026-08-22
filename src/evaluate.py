"""Evaluate a fine-tuned run's checkpoint on the held-out test set.

Intended to be invoked once per run from notebooks/kaggle_training.ipynb, e.g.:

    python src/evaluate.py --run_name combined_25pct --checkpoint_dir results/checkpoints/combined_25pct

Generates summaries on data/processed/test.jsonl with the checkpoint from
train.py, scores them with ROUGE-1/2/L (config["evaluation"]["metrics"]),
and logs the results via utils.ExperimentLogger so every run lands in
results/metrics/all_runs_summary.csv for cross-run comparison.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils import PROJECT_ROOT, ExperimentLogger, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUMMARIZE_PREFIX = "summarize: "


def _import_hf_evaluate():
    """Import the pip `evaluate` package without self-shadowing.

    Running `python src/evaluate.py` puts this file's own directory first on
    sys.path, so a plain `import evaluate` resolves to this file itself
    (also named evaluate.py) instead of the installed HF `evaluate` library.
    Drop this file's directory from sys.path for the duration of the import.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    sys.modules.pop("evaluate", None)
    original_path = sys.path
    sys.path = [p for p in original_path if os.path.abspath(p or os.getcwd()) != src_dir]
    try:
        import evaluate as hf_evaluate_module
    finally:
        sys.path = original_path
    return hf_evaluate_module


hf_evaluate = _import_hf_evaluate()


def load_test_set(processed_dir: str) -> list[dict]:
    """Load the held-out test split.

    Args:
        processed_dir: Directory holding data/processed/test.jsonl.

    Returns:
        List of {id, article, summary} dicts to evaluate against.
    """
    path = Path(processed_dir) / "test.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_summaries(test_examples: list[dict], checkpoint_dir: str, config: dict) -> list[str]:
    """Generate a summary for each test article using the fine-tuned checkpoint.

    Uses config["evaluation"]["generation_max_length"] and
    config["evaluation"]["generation_num_beams"] for beam-search decoding.

    Args:
        test_examples: Test examples from load_test_set().
        checkpoint_dir: Path to the fine-tuned model checkpoint, e.g.
            results/checkpoints/<run_name>/.
        config: Full project config, as returned by utils.load_config().

    Returns:
        A list of generated summary strings, aligned with `test_examples`.
    """
    eval_cfg = config["evaluation"]
    model_cfg = config["model"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
    model.eval()

    predictions = []
    for example in test_examples:
        inputs = tokenizer(
            SUMMARIZE_PREFIX + example["article"],
            max_length=model_cfg["max_input_length"],
            truncation=True,
            return_tensors="pt",
        )
        output_ids = model.generate(
            **inputs,
            max_length=eval_cfg["generation_max_length"],
            num_beams=eval_cfg["generation_num_beams"],
        )
        predictions.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    return predictions


def compute_rouge(predictions: list[str], references: list[str], metrics: list[str]) -> dict[str, float]:
    """Score generated summaries against references with ROUGE.

    Args:
        predictions: Generated summaries from generate_summaries().
        references: Ground-truth summaries, aligned with `predictions`.
        metrics: Which ROUGE variants to compute, config["evaluation"]["metrics"]
            (e.g. ["rouge1", "rouge2", "rougeL"]).

    Returns:
        Dict mapping each metric name to its aggregate F-measure score.
    """
    rouge = hf_evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references, rouge_types=metrics)
    return {metric: scores[metric] for metric in metrics}


def evaluate_checkpoint(checkpoint_path: str, config: dict) -> dict:
    """Run full ROUGE evaluation for one fine-tuned checkpoint against the test set.

    Args:
        checkpoint_path: Path to the fine-tuned model checkpoint.
        config: Full project config, as returned by utils.load_config().

    Returns:
        {"rouge1", "rouge2", "rougeL", "num_test_examples"}.
    """
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    test_examples = load_test_set(processed_dir)

    predictions = generate_summaries(test_examples, checkpoint_path, config)
    references = [example["summary"] for example in test_examples]

    rouge_scores = compute_rouge(predictions, references, config["evaluation"]["metrics"])

    return {
        **rouge_scores,
        "num_test_examples": len(test_examples),
    }


def run_evaluation(run_name: str, checkpoint_dir: str, config_path: str = None) -> None:
    """Run full evaluation for one run and log ROUGE scores via ExperimentLogger.

    Args:
        run_name: One of the 9 experiment matrix run names, must match the
            run_name used in train.py so metrics land in the same record.
        checkpoint_dir: Path to the fine-tuned model checkpoint for this run.
        config_path: Optional override for the config file path.
    """
    config = load_config(config_path)

    exp_logger = ExperimentLogger(run_name, config)
    exp_logger.start_timer("evaluation")
    result = evaluate_checkpoint(checkpoint_dir, config)
    exp_logger.stop_timer("evaluation")

    for key, value in result.items():
        exp_logger.log_metric(key, value)

    exp_logger.save()
    logger.info("Run '%s' evaluation complete: %s", run_name, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned mT5-small run with ROUGE.")
    parser.add_argument("--run_name", required=True, help="Run name, e.g. combined_25pct")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to the fine-tuned checkpoint")
    parser.add_argument("--config", default=None, help="Optional path to experiment_config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.run_name, args.checkpoint_dir, args.config)
