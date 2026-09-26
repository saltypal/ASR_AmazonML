from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .pipeline import (
    candidate_stage,
    feature_stage,
    inference_stage,
    preprocess_stage,
    run_all,
    run_official_validator,
    train_stage,
)
from .runtime import RuntimeBudget, build_run_metadata


def _resolved_arguments(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    repo_root = args.repo_root.resolve()
    data_root = args.data_root.resolve()
    work_root = args.work_dir.resolve()
    output_root = args.output_dir.resolve()
    config = load_config(args.config)
    config["_data_root"] = str(data_root)
    return repo_root, data_root, work_root, output_root, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ber-pipeline",
        description="Multilingual business entity-resolution pipeline.",
    )
    parser.add_argument(
        "stage",
        choices=("inspect", "preprocess", "candidates", "features", "train", "infer", "validate", "run-all"),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "test"), help="Required for candidates/features.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root, data_root, work_root, output_root, config = _resolved_arguments(args)
    if args.stage == "inspect":
        payload = {
            "repo_root": str(repo_root),
            "data_root": str(data_root),
            "work_dir": str(work_root),
            "output_dir": str(output_root),
            "config": config,
            "runtime": build_run_metadata(repo_root, config),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.stage in {"candidates", "features"} and not args.split:
        raise SystemExit(f"--split is required for the {args.stage} stage.")

    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    budget = RuntimeBudget(
        float(config["project"]["time_budget_minutes"]),
        float(config["project"]["output_reserve_minutes"]),
    )
    if args.stage == "preprocess":
        preprocess_stage(data_root, work_root, config)
    elif args.stage == "candidates":
        candidate_stage(work_root, config, args.split, budget)
    elif args.stage == "features":
        feature_stage(data_root, work_root, config, args.split)
    elif args.stage == "train":
        train_stage(data_root, work_root, config)
    elif args.stage == "infer":
        inference_stage(work_root, output_root)
    elif args.stage == "validate":
        run_official_validator(repo_root, data_root, output_root)
    elif args.stage == "run-all":
        result = run_all(repo_root, data_root, work_root, output_root, config)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
