#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import logging
import sys
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ae_depth.benchmark import run_benchmark  # noqa: E402


class _ClientTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        value = datetime.datetime.fromtimestamp(
            record.created, tz=ZoneInfo("Asia/Shanghai")
        )
        return value.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


def create_logger(output_dir: Path) -> tuple[logging.Logger, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
        "%Y%m%d-%H%M%S"
    )
    log_path = output_dir / f"eval_{timestamp}.log"
    logger = logging.getLogger("ae_depth.eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = _ClientTimeFormatter("%(asctime)s  %(levelname)5s  %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger, log_path


def parse_config():
    parser = argparse.ArgumentParser(description="AE depth reconstruction evaluator")
    parser.add_argument(
        "--cfg_file",
        "--config",
        dest="cfg_file",
        type=Path,
        default=PROJECT_ROOT / "cfgs/cosmos_ci16x16.yaml",
        help="evaluation YAML config",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help="override BENCHMARK.BATCH_SIZE",
    )
    parser.add_argument(
        "--limit_per_dataset",
        "--limit-per-dataset",
        dest="limit_per_dataset",
        type=int,
        default=0,
        help="evenly sample at most N entries per dataset; 0 uses full lists",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    with args.cfg_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.output_dir is not None:
        config["BENCHMARK"]["OUTPUT_DIR"] = str(args.output_dir.resolve())
    if args.batch_size is not None:
        config["BENCHMARK"]["BATCH_SIZE"] = args.batch_size
    return args, config


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args, config = parse_config()
    output_dir = Path(config["BENCHMARK"]["OUTPUT_DIR"])
    logger, log_path = create_logger(output_dir)
    logger.info("---------------- AE_depth evaluation ----------------")
    for key, value in vars(args).items():
        logger.info("%-20s %s", key, value)
    logger.info("log_file             %s", log_path)
    logger.info("---------------- Configuration ----------------")
    for line in yaml.safe_dump(config, allow_unicode=True, sort_keys=False).splitlines():
        logger.info(line)
    run_benchmark(
        config,
        limit_per_dataset=args.limit_per_dataset,
        batch_size_override=args.batch_size,
        logger=logger,
    )


if __name__ == "__main__":
    main()
