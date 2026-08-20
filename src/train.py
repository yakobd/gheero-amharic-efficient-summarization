"""Fine-tune mT5-small on a single subset (one run of the 9-run experiment matrix).

Intended to be invoked once per run from notebooks/kaggle_training.ipynb, e.g.:

    python src/train.py --run_name combined_25pct --subset_path data/subsets/combined_25pct.jsonl

Trains config["model"]["base_model"] with the hyperparameters in
config["training"], logs training time / data size / steps / peak GPU memory
via utils.ExperimentLogger, and saves the fine-tuned model for evaluate.py to
score.

Scaffolding only for Phase 3 — implemented in Phase 4/5.
"""

import argparse
from pathlib import Path


def load_subset(subset_path: str) -> list[dict]:
    """Load a JSONL training subset produced by build_subsets.py.

    Args:
        subset_path: Path to data/subsets/<run_name>.jsonl.

    Returns:
        List of {id, article, summary} dicts to train on.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


def build_trainer(train_examples: list[dict], val_examples: list[dict], config: dict, output_dir: Path):
    """Construct the Seq2SeqTrainer (model, tokenizer, data collator, TrainingArguments).

    Args:
        train_examples: Training subset for this run.
        val_examples: Validation examples from data/processed/validation.jsonl.
        config: Full project config, as returned by utils.load_config().
        output_dir: Directory to save checkpoints to, results/checkpoints/<run_name>/.

    Returns:
        A configured Seq2SeqTrainer ready to call .train() on.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


def run_training(run_name: str, subset_path: str, config_path: str = None) -> None:
    """Run one full training run: load data, train, and log timing/efficiency metrics.

    Records training time (ExperimentLogger.start_timer/stop_timer), data
    size, step count, and peak GPU memory for this run, then saves the
    result via ExperimentLogger.save() so it feeds results/metrics/all_runs_summary.csv.

    Args:
        run_name: One of the 9 experiment matrix run names, e.g. "quality_only_50pct".
        subset_path: Path to the training subset JSONL for this run.
        config_path: Optional override for the config file path.
    """
    raise NotImplementedError("Implemented in Phase 4/5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune mT5-small on one experiment subset.")
    parser.add_argument("--run_name", required=True, help="Run name, e.g. combined_25pct")
    parser.add_argument("--subset_path", required=True, help="Path to the subset JSONL file")
    parser.add_argument("--config", default=None, help="Optional path to experiment_config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.run_name, args.subset_path, args.config)
