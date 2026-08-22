"""Fine-tune mT5-small on a single subset (one run of the 9-run experiment matrix).

Intended to be invoked once per run from notebooks/kaggle_training.ipynb, e.g.:

    python src/train.py --run_name combined_25pct --subset_path data/subsets/combined_25pct.jsonl

Trains config["model"]["base_model"] with the hyperparameters in
config["training"], logs training time / data size / steps / peak GPU memory
via utils.ExperimentLogger, and saves the fine-tuned model for evaluate.py to
score.
"""

import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from utils import PROJECT_ROOT, ExperimentLogger, load_config, set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUMMARIZE_PREFIX = "summarize: "
DEFAULT_LOGGING_STEPS = 50


def train_one_run(run_name: str, subset_path: str, config: dict) -> dict:
    """Fine-tune config["model"]["base_model"] on one experiment-matrix subset.

    Args:
        run_name: Run name, e.g. "combined_25pct" -- used to name the checkpoint dir.
        subset_path: Path to the training subset JSONL for this run.
        config: Full project config, as returned by utils.load_config().

    Returns:
        {"num_train_examples", "num_epochs_completed", "peak_gpu_mem_mb"}.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_cfg = config["model"]
    train_cfg = config["training"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["base_model"])

    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": str(subset_path),
            "validation": str(processed_dir / "validation.jsonl"),
        },
    )

    def preprocess(examples: dict) -> dict:
        inputs = [SUMMARIZE_PREFIX + article for article in examples["article"]]
        model_inputs = tokenizer(
            inputs,
            max_length=model_cfg["max_input_length"],
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=examples["summary"],
            max_length=model_cfg["max_target_length"],
            truncation=True,
            # no padding here -- DataCollatorForSeq2Seq pads dynamically
            # per-batch and correctly masks label padding with -100
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw_datasets.map(
        preprocess, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    output_dir = PROJECT_ROOT / "results" / "checkpoints" / run_name

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        fp16=train_cfg["fp16"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        eval_strategy=train_cfg["eval_strategy"],
        logging_steps=DEFAULT_LOGGING_STEPS,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_result = trainer.train()

    peak_gpu_mem_mb = None
    if torch.cuda.is_available():
        peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return {
        "num_train_examples": len(tokenized["train"]),
        "num_epochs_completed": train_result.metrics.get("epoch", train_cfg["num_train_epochs"]),
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
    }


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
    config = load_config(config_path)
    set_global_seed(config["project"]["seed"])

    exp_logger = ExperimentLogger(run_name, config)
    exp_logger.start_timer("training")
    result = train_one_run(run_name, subset_path, config)
    exp_logger.stop_timer("training")

    for key, value in result.items():
        exp_logger.log_metric(key, value)

    exp_logger.save()
    logger.info("Run '%s' complete: %s", run_name, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune mT5-small on one experiment subset.")
    parser.add_argument("--run_name", required=True, help="Run name, e.g. combined_25pct")
    parser.add_argument("--subset_path", required=True, help="Path to the subset JSONL file")
    parser.add_argument("--config", default=None, help="Optional path to experiment_config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.run_name, args.subset_path, args.config)
