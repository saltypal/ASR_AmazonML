from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def detect_environment() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return "kaggle"
    if os.environ.get("SM_TRAINING_ENV") or Path("/opt/ml").exists():
        return "sagemaker"
    if os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    return "local"


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass
class RuntimeBudget:
    total_minutes: float
    reserve_minutes: float
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed_minutes(self) -> float:
        return (time.monotonic() - self.started_at) / 60.0

    @property
    def remaining_minutes(self) -> float:
        return max(0.0, self.total_minutes - self.elapsed_minutes)

    @property
    def optional_minutes(self) -> float:
        return max(0.0, self.remaining_minutes - self.reserve_minutes)

    def require(self, estimated_minutes: float, stage: str, optional: bool = False) -> None:
        available = self.optional_minutes if optional else self.remaining_minutes
        if estimated_minutes > available:
            kind = "optional" if optional else "required"
            raise TimeoutError(
                f"Skipping {kind} stage '{stage}': needs about {estimated_minutes:.1f} minutes, "
                f"but only {available:.1f} are available."
            )


def build_run_metadata(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": detect_environment(),
        "git_commit": git_commit(repo_root),
        "config_hash": config.get("_config_hash", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": config["project"]["seed"],
    }
    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = torch.cuda.is_available()
        metadata["gpu_count"] = torch.cuda.device_count()
        metadata["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        metadata["torch"] = None
        metadata["cuda_available"] = False
        metadata["gpu_count"] = 0
        metadata["gpus"] = []
    return metadata


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
