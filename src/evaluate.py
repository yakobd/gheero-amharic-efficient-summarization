"""Evaluate a fine-tuned run's checkpoint on the held-out test set.

Intended to be invoked once per run from notebooks/kaggle_training.ipynb, e.g.:

    python src/evaluate.py --run_name combined_25pct --checkpoint_dir results/checkpoints/combined_25pct

Generates summaries on data/processed/test.jsonl with the checkpoint from
train.py, scores them with ROUGE-1/2/L (config["evaluation"]["metrics"]),
and logs the results via utils.ExperimentLogger so every run lands in
results/metrics/all_runs_summary.csv for cross-run comparison.

Scaffolding only for Phase 3 — implemented in Phase 4/5.
"""

import argparse


def load_test_set(processed_dir: str) -> list[dict]:
    """Load the held-out test split.

    Args:
        processed_dir: Directory holding data/processed/test.jsonl.

    Returns:
        List of {id, article, summary} dicts to evaluate against.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


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
    raise NotImplementedError("Implemented in Phase 4/5")


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
    raise NotImplementedError("Implemented in Phase 4/5")


def run_evaluation(run_name: str, checkpoint_dir: str, config_path: str = None) -> None:
    """Run full evaluation for one run and log ROUGE scores via ExperimentLogger.

    Args:
        run_name: One of the 9 experiment matrix run names, must match the
            run_name used in train.py so metrics land in the same record.
        checkpoint_dir: Path to the fine-tuned model checkpoint for this run.
        config_path: Optional override for the config file path.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned mT5-small run with ROUGE.")
    parser.add_argument("--run_name", required=True, help="Run name, e.g. combined_25pct")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to the fine-tuned checkpoint")
    parser.add_argument("--config", default=None, help="Optional path to experiment_config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.run_name, args.checkpoint_dir, args.config)
