"""Shared utilities: config loading, global seeding, and experiment logging.

Every src/ script pulls its parameters from configs/experiment_config.yaml
through load_config() so the config file stays the single source of truth
for the project.
"""

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_config.yaml"


def load_config(config_path: Optional[str] = None) -> dict:
    """Load the project YAML config into a nested dict.

    Args:
        config_path: Path to the config file. Defaults to
            configs/experiment_config.yaml at the project root.

    Returns:
        Parsed config as a dict, keyed by section (project, data, model,
        quality_scoring, diversity_clustering, experiment_matrix,
        training, evaluation, paths).
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_global_seed(seed: int) -> None:
    """Seed python's random, numpy, and torch (if installed) for reproducibility.

    Args:
        seed: Seed value, normally config["project"]["seed"].
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class ExperimentLogger:
    """Tracks timing and metrics for a single experiment run.

    Writes a full per-run record to results/metrics/<run_name>.json and
    appends a summary row to results/metrics/all_runs_summary.csv, so
    individual runs can be inspected in detail while the CSV gives a
    quick cross-run comparison for the 9-run experiment matrix.
    """

    def __init__(self, run_name: str, config: dict, metrics_dir: Optional[str] = None):
        self.run_name = run_name
        self.config = config
        self.metrics_dir = Path(metrics_dir) if metrics_dir else PROJECT_ROOT / config["paths"]["results_metrics"]
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.metrics: dict[str, Any] = {}
        self._timers: dict[str, float] = {}
        self._durations: dict[str, float] = {}

    def start_timer(self, name: str = "total") -> None:
        """Start (or restart) a named timer, e.g. 'total' or 'training'."""
        self._timers[name] = time.time()

    def stop_timer(self, name: str = "total") -> float:
        """Stop a named timer and record its elapsed duration in seconds."""
        if name not in self._timers:
            raise ValueError(f"Timer '{name}' was never started")
        elapsed = time.time() - self._timers.pop(name)
        self._durations[name] = elapsed
        return elapsed

    def log_metric(self, key: str, value: Any) -> None:
        """Record a single metric (e.g. rouge1, peak_gpu_memory_mb, data_size)."""
        self.metrics[key] = value

    def save(self) -> None:
        """Write the full run record to JSON and append a row to the summary CSV."""
        record = {
            "run_name": self.run_name,
            "config": self.config,
            "metrics": self.metrics,
            "durations_seconds": self._durations,
        }

        json_path = self.metrics_dir / f"{self.run_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        self._append_summary_row()

    def _append_summary_row(self) -> None:
        summary_path = self.metrics_dir / "all_runs_summary.csv"
        row = {"run_name": self.run_name, **self.metrics, **self._durations}

        file_exists = summary_path.exists()
        existing_fieldnames: list[str] = []
        if file_exists:
            with open(summary_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_fieldnames = next(reader, [])

        fieldnames = list(existing_fieldnames) if existing_fieldnames else []
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

        needs_rewrite = file_exists and fieldnames != existing_fieldnames
        rows_to_keep = []
        if needs_rewrite:
            with open(summary_path, "r", newline="", encoding="utf-8") as f:
                rows_to_keep = list(csv.DictReader(f))

        mode = "w" if (not file_exists or needs_rewrite) else "a"
        with open(summary_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
                for existing_row in rows_to_keep:
                    writer.writerow(existing_row)
            writer.writerow(row)
