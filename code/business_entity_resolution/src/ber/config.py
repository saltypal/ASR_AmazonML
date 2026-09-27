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
        "train_queries_per_country": None,
    },
    "candidate_generation": {
        "method": "char_tfidf",
        "token_top_k": 12,
        "token_address_top_k": 0,
        "token_address_max_document_frequency": 20_000,
        "token_min_similarity": 0.01,
        "token_hash_features": 4_194_304,
        "token_query_chunk_size": 10_000,
        "token_threads": 4,
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
        "tfidf_query_chunk_size": 2_000,
        "max_tfidf_document_frequency": 5_000,
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


def _reject_unknown_keys(
    supplied: dict[str, Any], reference: dict[str, Any], prefix: str = ""
) -> None:
    for key, value in supplied.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in reference:
            raise ValueError(f"Unknown configuration key: {path}")
        if isinstance(value, dict):
            expected = reference[key]
            if not isinstance(expected, dict):
                raise ValueError(f"Configuration value {path} must not be a mapping.")
            _reject_unknown_keys(value, expected, path)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    if not isinstance(supplied, dict):
        raise ValueError("The configuration root must be a mapping.")
    _reject_unknown_keys(supplied, DEFAULT_CONFIG)
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
    embeddings = config["embeddings"]
    training = config["training"]
    if project["time_budget_minutes"] <= 0:
        raise ValueError("time_budget_minutes must be positive.")
    if project["output_reserve_minutes"] < 0:
        raise ValueError("output_reserve_minutes cannot be negative.")
    if project["time_budget_minutes"] <= project["output_reserve_minutes"]:
        raise ValueError("The total runtime must exceed the reserved output time.")
    if project["chunksize"] <= 0:
        raise ValueError("chunksize must be positive.")
    if project["feature_chunk_rows"] <= 0:
        raise ValueError("feature_chunk_rows must be positive.")
    sample_rows = project.get("sample_rows_per_file")
    if sample_rows is not None and sample_rows <= 0:
        raise ValueError("sample_rows_per_file must be null or positive.")
    train_queries = project.get("train_queries_per_country")
    if train_queries is not None and train_queries <= 0:
        raise ValueError("train_queries_per_country must be null or positive.")
    if candidates["method"] not in {"char_tfidf", "token_hash"}:
        raise ValueError("candidate_generation.method must be char_tfidf or token_hash.")
    for key in ("token_top_k", "token_hash_features", "token_query_chunk_size", "token_threads"):
        if candidates[key] <= 0:
            raise ValueError(f"candidate_generation.{key} must be positive.")
    if candidates["token_address_top_k"] < 0:
        raise ValueError("candidate_generation.token_address_top_k cannot be negative.")
    if candidates["token_address_max_document_frequency"] <= 0:
        raise ValueError("candidate_generation.token_address_max_document_frequency must be positive.")
    if not 0 <= candidates["token_min_similarity"] <= 1:
        raise ValueError("token_min_similarity must be between 0 and 1.")
    if not 1 <= candidates["max_candidates_per_query"] <= 500:
        raise ValueError("max_candidates_per_query must be between 1 and 500.")
    if candidates["name_top_k"] < 0 or candidates["address_top_k"] < 0:
        raise ValueError("Sparse candidate top-K values cannot be negative.")
    if candidates["name_top_k"] + candidates["address_top_k"] <= 0:
        raise ValueError("At least one sparse candidate lane must be enabled.")
    if candidates["semantic_pairs_per_query"] < 0:
        raise ValueError("semantic_pairs_per_query cannot be negative.")
    if candidates["max_exact_block_size"] <= 0:
        raise ValueError("max_exact_block_size must be positive.")
    if not 1 <= candidates["char_ngram_min"] <= candidates["char_ngram_max"]:
        raise ValueError("Character n-gram bounds must satisfy 1 <= min <= max.")
    for key in ("min_name_similarity", "min_address_similarity"):
        if not 0 <= candidates[key] <= 1:
            raise ValueError(f"{key} must be between 0 and 1.")
    if candidates["max_tfidf_features"] <= 0:
        raise ValueError("max_tfidf_features must be positive.")
    if candidates["tfidf_query_chunk_size"] <= 0:
        raise ValueError("tfidf_query_chunk_size must be positive.")
    if candidates["max_tfidf_document_frequency"] <= 0:
        raise ValueError("max_tfidf_document_frequency must be positive.")
    for key in ("max_length", "batch_size_per_gpu", "max_minutes"):
        if embeddings[key] <= 0:
            raise ValueError(f"embeddings.{key} must be positive.")
    if not 0 < training["validation_fraction"] < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")
    if not 0 <= training["holdout_fraction"] < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5.")
    validation_total = training["validation_fraction"] + training["holdout_fraction"]
    if not 0 < validation_total < 0.5:
        raise ValueError("Validation plus holdout fractions must be between 0 and 0.5.")
    if training["tuning_trials"] <= 0:
        raise ValueError("tuning_trials must be positive.")
    if training["tuning_timeout_seconds"] <= 0:
        raise ValueError("tuning_timeout_seconds must be positive.")
    if training["tuning_timeout_seconds"] > project["time_budget_minutes"] * 60:
        raise ValueError("Tuning timeout exceeds the complete run budget.")
    if training["tuning_sample_rows"] <= 0:
        raise ValueError("tuning_sample_rows must be positive.")
    if training["max_negatives_per_query"] < 1:
        raise ValueError("max_negatives_per_query must be positive.")
    if training["max_boost_rounds"] <= 0 or training["early_stopping_rounds"] <= 0:
        raise ValueError("Boosting and early-stopping rounds must be positive.")
    if not 0 <= training["threshold_min"] <= training["threshold_max"] <= 1:
        raise ValueError("Threshold bounds must satisfy 0 <= min <= max <= 1.")
    if training["threshold_steps"] < 2:
        raise ValueError("threshold_steps must be at least 2.")
    device = str(training["device"])
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("training.device must be 'cpu' or a CUDA device string.")
