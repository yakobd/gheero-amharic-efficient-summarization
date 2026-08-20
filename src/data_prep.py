"""Load XL-Sum Amharic, apply a word-count length filter, and save JSONL splits.

Uses XL-Sum's own predefined train/validation/test splits rather than
re-splitting the data, so results stay comparable to published XL-Sum
baselines. Run directly:

    python src/data_prep.py
"""

import json
import logging
from pathlib import Path

from datasets import load_dataset

from utils import PROJECT_ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def word_count(text: str) -> int:
    """Whitespace-based word count, used for the article length filter."""
    return len(text.split())


def filter_example(example: dict, min_words: int, max_words: int) -> bool:
    """Keep an example only if its article's word count falls within [min_words, max_words]."""
    n_words = word_count(example["text"])
    return min_words <= n_words <= max_words


def prepare_split(dataset_split, min_words: int, max_words: int) -> list[dict]:
    """Filter a raw XL-Sum split by article length and project it to {id, article, summary}."""
    records = []
    for example in dataset_split:
        if not filter_example(example, min_words, max_words):
            continue
        records.append(
            {
                "id": example["id"],
                "article": example["text"],
                "summary": example["summary"],
            }
        )
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    """Write a list of dicts to a JSONL file, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(config_path: str = None) -> None:
    config = load_config(config_path)
    data_cfg = config["data"]

    logger.info("Loading %s (%s) ...", data_cfg["dataset_name"], data_cfg["language_config"])
    dataset = load_dataset(data_cfg["dataset_name"], data_cfg["language_config"])

    processed_dir = PROJECT_ROOT / data_cfg["processed_dir"]
    min_words = data_cfg["min_article_words"]
    max_words = data_cfg["max_article_words"]

    split_map = {"train": "train", "validation": "validation", "test": "test"}
    for split_name, hf_split in split_map.items():
        logger.info("Processing split '%s' (raw size=%d) ...", split_name, len(dataset[hf_split]))
        records = prepare_split(dataset[hf_split], min_words, max_words)
        logger.info(
            "Split '%s': kept %d/%d examples after length filter [%d, %d] words",
            split_name,
            len(records),
            len(dataset[hf_split]),
            min_words,
            max_words,
        )

        out_path = processed_dir / f"{split_name}.jsonl"
        save_jsonl(records, out_path)
        logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
