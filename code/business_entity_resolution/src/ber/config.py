from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "seed": 42,
        "time_budget_minutes": 480,
        "output_reserve_minutes": 35,
        "chunksize": 250_000,
        "feature_chunk_rows": 400_000,
        "sample_rows_per_file": None,
    },
    "candidate_generation": {
        "name_top_k": 12,
        "address_top_k": 10,
        "semantic_pairs_per_query": 3,
        "max_candidates_per_query": 40,
        "max_exact_block_size": 200,
        "char_ngram_min": 3,
        "char_ngram_max": 5,
        "min_name_similarity": 0.18,
        "min_address_similarity": 0.16,
        "max_tfidf_features": 400_000,
    },
    "embeddings": {
        "enabled": True,
        "model_name": "intfloat/multilingual-e5-small",
        "model_path": None,
        "max_length": 128,
        "batch_size_per_gpu": 192,
        "cross_script_only": True,
        "max_minutes": 65,
    },
    "training": {
        "validation_fraction": 0.10,
        "holdout_fraction": 0.10,
        "tuning_trials": 12,
        "tuning_timeout_seconds": 2400,
        "tuning_sample_rows": 4_000_000,
        "max_negatives_per_query": 12,
        "early_stopping_rounds": 75,
        "max_boost_rounds": 1400,
        "threshold_min": 0.20,
        "threshold_max": 0.98,
        "threshold_steps": 79,
        "device": "cuda",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    config = _deep_merge(DEFAULT_CONFIG, supplied)
    validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    config["_config_hash"] = config_hash(config)
    return config


def config_hash(config: dict[str, Any]) -> str:
    material = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def validate_config(config: dict[str, Any]) -> None:
    project = config["project"]
    candidates = config["candidate_generation"]
    training = config["training"]
    if project["time_budget_minutes"] <= project["output_reserve_minutes"]:
        raise ValueError("The total runtime must exceed the reserved output time.")
    if project["chunksize"] <= 0:
        raise ValueError("chunksize must be positive.")
    if project["feature_chunk_rows"] <= 0:
        raise ValueError("feature_chunk_rows must be positive.")
    if not 1 <= candidates["max_candidates_per_query"] <= 500:
        raise ValueError("max_candidates_per_query must be between 1 and 500.")
    if candidates["name_top_k"] + candidates["address_top_k"] <= 0:
        raise ValueError("At least one sparse candidate lane must be enabled.")
    if candidates["semantic_pairs_per_query"] < 0:
        raise ValueError("semantic_pairs_per_query cannot be negative.")
    validation_total = training["validation_fraction"] + training["holdout_fraction"]
    if not 0 < validation_total < 0.5:
        raise ValueError("Validation plus holdout fractions must be between 0 and 0.5.")
    if training["tuning_timeout_seconds"] > project["time_budget_minutes"] * 60:
        raise ValueError("Tuning timeout exceeds the complete run budget.")
    if training["max_negatives_per_query"] < 1:
        raise ValueError("max_negatives_per_query must be positive.")
