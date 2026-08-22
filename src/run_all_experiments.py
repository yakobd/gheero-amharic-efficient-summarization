"""Run the full remaining experiment pipeline unattended, end to end.

Intended for Kaggle's "Save & Run All" (background execution), not
interactive cell-by-cell running: data prep -> quality scoring -> diversity
clustering -> subset building -> train + evaluate every currently-runnable
condition, freeing checkpoint disk space between runs and continuing past
any single run's failure. Run directly:

    python src/run_all_experiments.py

train.py and evaluate.py are invoked as separate subprocesses (not imported
as modules): each run then gets its own CUDA context that's fully released
on process exit -- covering Trainer-internal state (optimizer, accelerator)
that manual del/gc.collect()/empty_cache() cleanup can't reach -- and
evaluate.py's internal sys.modules juggling (to avoid self-shadowing the pip
`evaluate` package) stays contained to its own process instead of clobbering
this one.
"""

import csv
import gc
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

import build_subsets
import data_prep
import diversity_clustering
import quality_scoring
from utils import PROJECT_ROOT

RUNS = [
    ("full_dataset", "data/subsets/full.jsonl"),
    ("random_25pct", "data/subsets/random_25pct.jsonl"),
    ("combined_25pct", "data/subsets/combined_25pct.jsonl"),
    # TODO(Phase 5+): random_50pct, combined_50pct, quality_only_{25,50}pct,
    # diversity_only_{25,50}pct once build_subsets.py's quality_only /
    # diversity_only builders are implemented and scope reopens past 25%.
]

ERROR_LOG_PATH = PROJECT_ROOT / "results" / "logs" / "errors.log"
SUMMARY_CSV_PATH = PROJECT_ROOT / "results" / "metrics" / "all_runs_summary.csv"


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _log_error(run_name: str, exc: BaseException) -> None:
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"=== {_timestamp()} -- run '{run_name}' failed ===\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        f.write("\n")


def _delete_checkpoint(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.exists():
        print(f"No checkpoint dir found at {checkpoint_dir} to delete")
        return

    freed_bytes = _dir_size_bytes(checkpoint_dir)
    shutil.rmtree(checkpoint_dir)
    print(f"Deleted checkpoint dir {checkpoint_dir} (freed {freed_bytes / (1024 ** 2):.1f} MB)")


def _print_summary() -> None:
    print(f"\n=== FINAL SUMMARY ({SUMMARY_CSV_PATH}) ===")
    if not SUMMARY_CSV_PATH.exists():
        print(f"No summary file found at {SUMMARY_CSV_PATH}")
        return

    with open(SUMMARY_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        print("(empty)")
        return

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def _run_train(run_name: str, subset_path: str) -> None:
    subprocess.run(
        [sys.executable, "src/train.py", "--run_name", run_name, "--subset_path", subset_path],
        check=True,
    )


def _run_evaluate(run_name: str, checkpoint_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "src/evaluate.py", "--run_name", run_name, "--checkpoint_dir", str(checkpoint_dir)],
        check=True,
    )


def main() -> None:
    print(f"[{_timestamp()}] Running data_prep...")
    data_prep.main()

    print(f"[{_timestamp()}] Running quality_scoring...")
    quality_scoring.main()

    print(f"[{_timestamp()}] Running diversity_clustering...")
    diversity_clustering.main()

    print(f"[{_timestamp()}] Running build_subsets...")
    build_subsets.main()

    for run_name, subset_path in RUNS:
        print(f"=== STARTING RUN: {run_name} === [{_timestamp()}]")
        checkpoint_dir = PROJECT_ROOT / "results" / "checkpoints" / run_name

        try:
            # check=True raises subprocess.CalledProcessError on a non-zero
            # exit; each run is its own process either way, so GPU memory is
            # fully released when it exits, whether the run succeeds or fails.
            _run_train(run_name, subset_path)
            time.sleep(15)
            _run_evaluate(run_name, checkpoint_dir)
            time.sleep(15)
            _delete_checkpoint(checkpoint_dir)
            print(f"=== COMPLETED RUN: {run_name} === [{_timestamp()}]")
        except Exception as exc:
            print(f"!!! RUN FAILED: {run_name} -- see {ERROR_LOG_PATH} !!! [{_timestamp()}]")
            print(traceback.format_exc())
            _log_error(run_name, exc)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

    _print_summary()


if __name__ == "__main__":
    main()
