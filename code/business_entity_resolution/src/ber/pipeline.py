from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .candidates import candidate_recall, combine_candidate_lanes, generate_candidates_for_source
from .embeddings import launch_parallel_encoding, load_embeddings
from .features import FEATURE_COLUMNS, build_pair_features, label_pair_features
from .io import (
    available_country_slugs,
    dataset_files,
    ground_truth_sets,
    load_country_partition,
    load_ground_truth,
    preprocess_dataset,
)
from .metrics import macro_fbeta, predictions_at_threshold
from .model import load_model_bundle, save_model_bundle, score_frame, tune_model
from .runtime import RuntimeBudget, build_run_metadata, write_json
from .splitting import assign_group_splits
from .submission import (
    combine_output_parts,
    validate_output_relationships,
    write_country_output_parts,
)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _reset_child(path: Path, parent: Path) -> None:
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    if resolved_path == resolved_parent or resolved_parent not in resolved_path.parents:
        raise ValueError(f"Refusing to reset unsafe path: {resolved_path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _model_reference(config: dict[str, Any]) -> str:
    configured_path = config["embeddings"].get("model_path")
    return str(configured_path or config["embeddings"]["model_name"])


def _gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def _semantic_rerank_lane(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    classical_pairs: pd.DataFrame,
    transient_root: Path,
    split: str,
    country_slug: str,
    target_source: str,
    config: dict[str, Any],
    timeout_seconds: float,
) -> pd.DataFrame:
    embedding_config = config["embeddings"]
    gpu_count = _gpu_count()
    if gpu_count < 1:
        raise RuntimeError(
            "Embeddings are enabled but no CUDA GPU is visible. Use smoke.yaml for CPU, "
            "or attach a Kaggle/AWS GPU."
        )
    if classical_pairs.empty:
        return pd.DataFrame()
    pair_view = classical_pairs.merge(
        queries[["entity_id", "name_script"]].rename(
            columns={"entity_id": "source1_entity_id", "name_script": "query_script"}
        ),
        on="source1_entity_id",
        how="left",
        validate="many_to_one",
    ).merge(
        candidates[["entity_id", "name_script"]].rename(
            columns={"entity_id": "candidate_entity_id", "name_script": "candidate_script"}
        ),
        on="candidate_entity_id",
        how="left",
        validate="many_to_one",
    )
    if bool(embedding_config["cross_script_only"]):
        eligible = pair_view[
            pair_view["query_script"].ne(pair_view["candidate_script"])
            & pair_view["query_script"].ne("NONE")
            & pair_view["candidate_script"].ne("NONE")
        ]
    else:
        eligible = pair_view
    eligible = eligible.sort_values(
        ["source1_entity_id", "retrieval_score"], ascending=[True, False], kind="stable"
    ).groupby("source1_entity_id", sort=False).head(
        int(config["candidate_generation"]["semantic_pairs_per_query"])
    )
    if eligible.empty:
        return pd.DataFrame()

    query_subset = queries[queries["entity_id"].isin(set(eligible["source1_entity_id"]))]
    candidate_subset = candidates[
        candidates["entity_id"].isin(set(eligible["candidate_entity_id"]))
    ]
    country_root = transient_root / split / country_slug / target_source
    _reset_child(country_root, transient_root)
    input_root = country_root / "input"
    query_input = input_root / "source1"
    candidate_input = input_root / target_source
    _reset_child(query_input, country_root)
    _reset_child(candidate_input, country_root)
    _atomic_parquet(query_subset, query_input / "part-00000.parquet")
    _atomic_parquet(candidate_subset, candidate_input / "part-00000.parquet")
    query_output = country_root / "output" / "source1"
    candidate_output = country_root / "output" / target_source
    model_reference = _model_reference(config)
    common = {
        "model_name": model_reference,
        "batch_size_per_gpu": int(embedding_config["batch_size_per_gpu"]),
        "max_length": int(embedding_config["max_length"]),
        "gpu_count": gpu_count,
        "non_latin_only": False,
    }
    deadline = time.monotonic() + timeout_seconds
    if not list(query_output.glob("_SUCCESS.rank-*")):
        launch_parallel_encoding(
            query_input,
            query_output,
            prefix="query",
            timeout_seconds=max(1.0, deadline - time.monotonic()),
            **common,
        )
    if not list(candidate_output.glob("_SUCCESS.rank-*")):
        launch_parallel_encoding(
            candidate_input,
            candidate_output,
            prefix="passage",
            timeout_seconds=max(1.0, deadline - time.monotonic()),
            **common,
        )
    query_ids, query_vectors = load_embeddings(query_output)
    candidate_ids, candidate_vectors = load_embeddings(candidate_output)
    query_positions = {entity_id: position for position, entity_id in enumerate(query_ids)}
    candidate_positions = {entity_id: position for position, entity_id in enumerate(candidate_ids)}
    query_index = eligible["source1_entity_id"].map(query_positions).to_numpy(dtype=np.int64)
    candidate_index = eligible["candidate_entity_id"].map(candidate_positions).to_numpy(dtype=np.int64)
    scores = np.einsum(
        "ij,ij->i", query_vectors[query_index], candidate_vectors[candidate_index]
    ).astype(np.float32)
    return pd.DataFrame(
        {
            "source1_entity_id": eligible["source1_entity_id"].to_numpy(),
            "candidate_entity_id": eligible["candidate_entity_id"].to_numpy(),
            "target_source": target_source.replace("source", "S"),
            "dense_score": scores,
        }
    )


def preprocess_stage(data_root: Path, work_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    destination = work_root / "preprocessed"
    _reset_child(destination, work_root)
    print("[preprocess] Streaming TSV files into normalized, country-partitioned Parquet.")
    return preprocess_dataset(data_root, destination, config)


def candidate_stage(
    work_root: Path,
    config: dict[str, Any],
    split: str,
    budget: RuntimeBudget,
    embedding_deadline: float | None = None,
) -> dict[str, Any]:
    preprocessed_root = work_root / "preprocessed"
    destination = work_root / "candidates" / split
    _reset_child(destination, work_root)
    transient_root = work_root / "transient_embeddings"
    candidate_config = config["candidate_generation"]
    embedding_config = config["embeddings"]
    dense_enabled = bool(
        embedding_config["enabled"]
        and candidate_config["semantic_pairs_per_query"] > 0
    )
    if embedding_deadline is None:
        embedding_deadline = time.monotonic() + float(embedding_config["max_minutes"]) * 60.0
    truth_frame = None
    if split == "train":
        nrows = config["project"].get("sample_rows_per_file")
        truth_frame = load_ground_truth(
            dataset_files(Path(config["_data_root"]), "train")["ground_truth"], nrows
        )
    diagnostics: list[dict[str, Any]] = []
    required_deadline = (
        budget.started_at
        + (budget.total_minutes - budget.reserve_minutes) * 60.0
    )

    for country_slug in available_country_slugs(preprocessed_root, split):
        budget.require(config["project"]["output_reserve_minutes"], f"{split} candidates")
        queries = load_country_partition(preprocessed_root, split, "source1", country_slug)
        if split == "train" and config["project"].get("train_queries_per_country") is not None:
            limit = int(config["project"]["train_queries_per_country"])
            queries = queries.sample(
                n=min(limit, len(queries)), random_state=int(config["project"]["seed"])
            ).sort_index().reset_index(drop=True)
            print(
                f"[candidates] {split}/{country_slug}: sampled {len(queries):,} "
                "Source-1 entities for supervised training",
                flush=True,
            )
        _atomic_parquet(
            queries[["entity_id"]], destination / f"query-ids-{country_slug}.parquet"
        )
        country_lanes: list[pd.DataFrame] = []
        target_frames: dict[str, pd.DataFrame] = {}
        classical_by_target: dict[str, pd.DataFrame] = {}
        for target_source, target_label in (("source2", "S2"), ("source3", "S3")):
            candidates = load_country_partition(preprocessed_root, split, target_source, country_slug)
            if dense_enabled:
                target_frames[target_source] = candidates
            if candidates.empty:
                continue
            print(
                f"[candidates] {split}/{country_slug}/{target_label}: "
                f"{len(queries):,} queries x {len(candidates):,} records"
            )
            classical = generate_candidates_for_source(
                queries,
                candidates,
                target_label,
                candidate_config,
                deadline=required_deadline,
            )
            if dense_enabled:
                classical_by_target[target_source] = classical
            country_lanes.append(classical)
            print(
                f"[candidates] {split}/{country_slug}/{target_label}: "
                f"retrieval complete | pairs={len(classical):,} | "
                f"budget_remaining_minutes={budget.remaining_minutes:.1f}",
                flush=True,
            )

        use_dense_here = dense_enabled and time.monotonic() < embedding_deadline
        if use_dense_here:
            budget.require(5.0, f"{split}/{country_slug} multilingual embeddings", optional=True)
            for target_source, classical in classical_by_target.items():
                if target_frames[target_source].empty:
                    continue
                seconds_left = min(
                    embedding_deadline - time.monotonic(), budget.optional_minutes * 60.0
                )
                if seconds_left <= 1:
                    break
                try:
                    country_lanes.append(
                        _semantic_rerank_lane(
                            queries,
                            target_frames[target_source],
                            classical,
                            transient_root,
                            split,
                            country_slug,
                            target_source,
                            config,
                            seconds_left,
                        )
                    )
                except TimeoutError:
                    print("[candidates] Multilingual reranking reached its time cap; continuing.")
                    break
        elif dense_enabled:
            print(
                f"[candidates] Embedding branch reached its {embedding_config['max_minutes']} minute cap; "
                "continuing with exact and character TF-IDF lanes."
            )

        pairs = combine_candidate_lanes(
            country_lanes, int(candidate_config["max_candidates_per_query"])
        )
        output_path = destination / f"country={country_slug}.parquet"
        _atomic_parquet(pairs, output_path)
        diagnostic: dict[str, Any] = {
            "country": country_slug,
            "queries": len(queries),
            "candidate_pairs": len(pairs),
        }
        if truth_frame is not None:
            country_query_ids = set(queries["entity_id"])
            country_truth = ground_truth_sets(
                truth_frame[truth_frame["source1_entity_id"].isin(country_query_ids)]
            )
            diagnostic.update(candidate_recall(pairs, country_truth))
        diagnostics.append(diagnostic)
        transient_country = transient_root / split / country_slug
        if transient_country.exists():
            shutil.rmtree(transient_country)
        del queries, country_lanes, target_frames, classical_by_target, candidates
    summary = {"split": split, "countries": diagnostics}
    write_json(destination / "manifest.json", summary)
    return summary


def _downsample_training_negatives(
    frame: pd.DataFrame, max_negatives_per_query: int
) -> pd.DataFrame:
    positives = frame[frame["label"] == 1]
    negatives = frame[frame["label"] == 0].sort_values(
        ["source1_entity_id", "retrieval_rank_score"], ascending=[True, False], kind="stable"
    )
    negatives = negatives.groupby("source1_entity_id", sort=False).head(max_negatives_per_query)
    return pd.concat([positives, negatives], ignore_index=True).sort_values(
        ["source1_entity_id", "label", "retrieval_rank_score"],
        ascending=[True, False, False],
        kind="stable",
    )


def _query_aligned_chunks(frame: pd.DataFrame, target_rows: int):
    start = 0
    total = len(frame)
    while start < total:
        end = min(total, start + target_rows)
        if end < total:
            boundary_id = frame.iloc[end - 1]["source1_entity_id"]
            while end < total and frame.iloc[end]["source1_entity_id"] == boundary_id:
                end += 1
        yield frame.iloc[start:end].copy()
        start = end


def feature_stage(
    data_root: Path,
    work_root: Path,
    config: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    destination = work_root / "features" / split
    _reset_child(destination, work_root)
    preprocessed_root = work_root / "preprocessed"
    candidate_root = work_root / "candidates" / split
    truth: dict[str, set[str]] | None = None
    if split == "train":
        nrows = config["project"].get("sample_rows_per_file")
        truth_frame = load_ground_truth(dataset_files(data_root, "train")["ground_truth"], nrows)
        query_id_paths = sorted(candidate_root.glob("query-ids-*.parquet"))
        if query_id_paths:
            selected_ids = set().union(
                *(set(pd.read_parquet(path)["entity_id"]) for path in query_id_paths)
            )
            truth_frame = truth_frame[truth_frame["source1_entity_id"].isin(selected_ids)]
        truth = ground_truth_sets(truth_frame)
    summaries = []
    for pair_path in sorted(candidate_root.glob("country=*.parquet")):
        country_slug = pair_path.stem.split("=", 1)[1]
        pairs = pd.read_parquet(pair_path)
        queries = load_country_partition(preprocessed_root, split, "source1", country_slug)
        if split == "train" and config["project"].get("train_queries_per_country") is not None:
            queries = queries[queries["entity_id"].isin(set(pairs["source1_entity_id"]))]
        target_frames = {
            "S2": load_country_partition(preprocessed_root, split, "source2", country_slug),
            "S3": load_country_partition(preprocessed_root, split, "source3", country_slug),
        }
        country_destination = destination / f"country={country_slug}"
        country_destination.mkdir(parents=True, exist_ok=True)
        total_rows = total_positives = 0
        for part_number, pair_chunk in enumerate(
            _query_aligned_chunks(pairs, int(config["project"]["feature_chunk_rows"]))
        ):
            chunk_features = []
            for target_label in ("S2", "S3"):
                source_pairs = pair_chunk[pair_chunk["target_source"] == target_label]
                if source_pairs.empty:
                    continue
                chunk_features.append(
                    build_pair_features(source_pairs, queries, target_frames[target_label])
                )
            features = (
                pd.concat(chunk_features, ignore_index=True)
                if chunk_features
                else pd.DataFrame(
                    columns=[
                        "source1_entity_id",
                        "candidate_entity_id",
                        "target_source",
                        *FEATURE_COLUMNS,
                    ]
                )
            )
            if truth is not None:
                features = label_pair_features(features, truth)
                features["split"] = assign_group_splits(
                    features["source1_entity_id"],
                    float(config["training"]["validation_fraction"]),
                    float(config["training"]["holdout_fraction"]),
                    int(config["project"]["seed"]),
                )
                features = _downsample_training_negatives(
                    features, int(config["training"]["max_negatives_per_query"])
                )
            total_rows += len(features)
            total_positives += int(features["label"].sum()) if "label" in features else 0
            _atomic_parquet(
                features, country_destination / f"part-{part_number:05d}.parquet"
            )
            print(
                f"[features] {split}/{country_slug}: part={part_number + 1} | "
                f"rows={len(features):,} | cumulative_rows={total_rows:,}",
                flush=True,
            )
        summaries.append(
            {
                "country": country_slug,
                "rows": total_rows,
                "positives": total_positives if truth is not None else None,
            }
        )
        print(f"[features] {split}/{country_slug}: {total_rows:,} model rows")
    summary = {"split": split, "countries": summaries}
    write_json(destination / "manifest.json", summary)
    return summary


def _truth_by_split(
    truth: dict[str, set[str]], config: dict[str, Any]
) -> dict[str, dict[str, set[str]]]:
    ids = pd.Series(list(truth), dtype=str)
    labels = assign_group_splits(
        ids,
        float(config["training"]["validation_fraction"]),
        float(config["training"]["holdout_fraction"]),
        int(config["project"]["seed"]),
    )
    return {
        split: {source1_id: truth[source1_id] for source1_id in ids[labels == split]}
        for split in ("train", "validation", "holdout")
    }


def train_stage(data_root: Path, work_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    feature_paths = sorted((work_root / "features" / "train").glob("country=*/part-*.parquet"))
    if not feature_paths:
        raise FileNotFoundError("No training feature files were found.")
    frame = pd.concat((pd.read_parquet(path) for path in feature_paths), ignore_index=True)
    required = {"label", "split", *FEATURE_COLUMNS}
    if missing := required - set(frame.columns):
        raise ValueError(f"Training features are missing columns: {sorted(missing)}")
    partitions = {name: frame[frame["split"] == name].reset_index(drop=True) for name in ("train", "validation", "holdout")}
    for name in ("train", "validation"):
        if partitions[name].empty or partitions[name]["label"].nunique() < 2:
            raise ValueError(
                f"The {name} partition does not contain both classes. A row-limited TSV smoke sample "
                "is not statistically valid for training; use the synthetic integration test or full data."
            )
    truth_frame = load_ground_truth(
        dataset_files(data_root, "train")["ground_truth"],
        config["project"].get("sample_rows_per_file"),
    )
    query_id_paths = sorted((work_root / "candidates" / "train").glob("query-ids-*.parquet"))
    if query_id_paths:
        trained_query_ids = set().union(
            *(set(pd.read_parquet(path)["entity_id"]) for path in query_id_paths)
        )
        truth_frame = truth_frame[truth_frame["source1_entity_id"].isin(trained_query_ids)]
        print(
            f"[train] Ground truth restricted to {len(truth_frame):,} retrieved training entities.",
            flush=True,
        )
    truths = _truth_by_split(ground_truth_sets(truth_frame), config)
    training_config = dict(config["training"])
    requested_device = str(training_config["device"])
    if requested_device.startswith("cuda") and _gpu_count() < 1:
        print("[train] CUDA requested but unavailable; falling back to CPU for this run.")
        requested_device = "cpu"
    training_config["device"] = requested_device
    print(
        f"[train] Tuning on {len(partitions['train']):,} train and "
        f"{len(partitions['validation']):,} validation candidate pairs ({requested_device})."
    )
    model, tuning_result = tune_model(
        partitions["train"],
        partitions["validation"],
        truths["validation"],
        training_config,
        int(config["project"]["seed"]),
    )
    threshold = float(tuning_result["threshold"])
    holdout_score = None
    if not partitions["holdout"].empty and truths["holdout"]:
        holdout_scored = score_frame(model, partitions["holdout"])
        holdout_predictions = predictions_at_threshold(holdout_scored, threshold)
        holdout_score = macro_fbeta(truths["holdout"], holdout_predictions, beta=0.5)
    metadata = {
        **tuning_result,
        "holdout_macro_f0_5": holdout_score,
        "feature_columns": FEATURE_COLUMNS,
        "device": requested_device,
    }
    save_model_bundle(model, work_root / "model", metadata)
    print(
        f"[train] Validation macro F0.5={tuning_result['macro_f0_5']:.6f}; "
        f"holdout={holdout_score if holdout_score is not None else 'n/a'}; threshold={threshold:.3f}"
    )
    return metadata


def inference_stage(work_root: Path, output_root: Path) -> dict[str, Any]:
    model, metadata = load_model_bundle(work_root / "model")
    threshold = float(metadata["threshold"])
    parts_root = output_root / "parts"
    if parts_root.exists():
        shutil.rmtree(parts_root)
    feature_root = work_root / "features" / "test"
    candidate_root = work_root / "candidates" / "test"
    preprocessed_root = work_root / "preprocessed"
    summaries = []
    for country_directory in sorted(feature_root.glob("country=*")):
        country_slug = country_directory.name.split("=", 1)[1]
        pairs = pd.read_parquet(candidate_root / f"country={country_slug}.parquet")
        queries = load_country_partition(preprocessed_root, "test", "source1", country_slug)
        selected_parts = []
        scored_rows = selected_rows = 0
        minimum_score = maximum_score = None
        for part_number, feature_path in enumerate(
            sorted(country_directory.glob("part-*.parquet")), start=1
        ):
            features = pd.read_parquet(feature_path)
            if features.empty:
                continue
            scored_part = score_frame(model, features)
            scored_rows += len(scored_part)
            part_minimum = float(scored_part["score"].min())
            part_maximum = float(scored_part["score"].max())
            minimum_score = part_minimum if minimum_score is None else min(minimum_score, part_minimum)
            maximum_score = part_maximum if maximum_score is None else max(maximum_score, part_maximum)
            selected_part = scored_part[scored_part["score"] >= threshold]
            selected_rows += len(selected_part)
            if not selected_part.empty:
                selected_parts.append(selected_part)
            print(
                f"[infer] {country_slug}: part={part_number} | "
                f"scored={scored_rows:,} | selected={selected_rows:,}",
                flush=True,
            )
        scored = (
            pd.concat(selected_parts, ignore_index=True)
            if selected_parts
            else pd.DataFrame(
                columns=["source1_entity_id", "candidate_entity_id", "target_source", "score"]
            )
        )
        write_country_output_parts(
            queries["entity_id"], pairs, scored, threshold, output_root, country_slug
        )
        summaries.append(
            {
                "country": country_slug,
                "pairs": len(pairs),
                "scored": scored_rows,
                "selected": selected_rows,
                "minimum_score": minimum_score,
                "maximum_score": maximum_score,
            }
        )
        print(
            f"[infer] {country_slug}: scored {scored_rows:,}, "
            f"selected {summaries[-1]['selected']:,}, max={summaries[-1]['maximum_score']}"
        )
    matching_path, candidate_path = combine_output_parts(output_root)
    relationship_summary = validate_output_relationships(matching_path, candidate_path)
    summary = {"threshold": threshold, "countries": summaries, **relationship_summary}
    write_json(output_root / "inference_manifest.json", summary)
    return summary


def run_official_validator(repo_root: Path, data_root: Path, output_root: Path) -> None:
    validator = repo_root / "utils" / "validate_submission.py"
    command = [
        sys.executable,
        str(validator),
        "--matching",
        str(output_root / "matching_results.tsv"),
        "--candidate",
        str(output_root / "candidate_pairs.tsv"),
        "--test-dir",
        str(data_root / "test"),
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def run_all(
    repo_root: Path,
    data_root: Path,
    work_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    config["_data_root"] = str(data_root.resolve())
    budget = RuntimeBudget(
        float(config["project"]["time_budget_minutes"]),
        float(config["project"]["output_reserve_minutes"]),
    )
    metadata = build_run_metadata(repo_root, config)
    write_json(work_root / "run_metadata.json", metadata)
    stage_started = time.monotonic()
    print("\n========== STAGE 1/7: PREPROCESS DATA ==========" , flush=True)
    preprocess_stage(data_root, work_root, config)
    print(f"[stage] Preprocessing finished in {(time.monotonic() - stage_started) / 60:.1f} min", flush=True)
    embedding_minutes_per_split = float(config["embeddings"]["max_minutes"]) / 2.0
    stage_started = time.monotonic()
    print("\n========== STAGE 2/7: TRAIN CANDIDATES ==========" , flush=True)
    candidate_stage(
        work_root,
        config,
        "train",
        budget,
        embedding_deadline=time.monotonic() + embedding_minutes_per_split * 60.0,
    )
    print(f"[stage] Train candidates finished in {(time.monotonic() - stage_started) / 60:.1f} min", flush=True)
    stage_started = time.monotonic()
    print("\n========== STAGE 3/7: TEST CANDIDATES ==========" , flush=True)
    candidate_stage(
        work_root,
        config,
        "test",
        budget,
        embedding_deadline=time.monotonic() + embedding_minutes_per_split * 60.0,
    )
    print(f"[stage] Test candidates finished in {(time.monotonic() - stage_started) / 60:.1f} min", flush=True)
    stage_started = time.monotonic()
    print("\n========== STAGE 4/7: BUILD TRAIN FEATURES ==========" , flush=True)
    feature_stage(data_root, work_root, config, "train")
    print(f"[stage] Train features finished in {(time.monotonic() - stage_started) / 60:.1f} min", flush=True)
    stage_started = time.monotonic()
    print("\n========== STAGE 5/7: BUILD TEST FEATURES ==========" , flush=True)
    feature_stage(data_root, work_root, config, "test")
    print(f"[stage] Test features finished in {(time.monotonic() - stage_started) / 60:.1f} min", flush=True)
    stage_started = time.monotonic()
    print("\n========== STAGE 6/7: TUNE AND TRAIN XGBOOST ==========" , flush=True)
    training = train_stage(data_root, work_root, config)
    print(f"[stage] Model training finished in {(time.monotonic() - stage_started) / 60:.1f} min", flush=True)
    stage_started = time.monotonic()
    print("\n========== STAGE 7/7: INFER, VALIDATE, SAVE ==========" , flush=True)
    inference = inference_stage(work_root, output_root)
    run_official_validator(repo_root, data_root, output_root)
    print(f"[stage] Inference and validation finished in {(time.monotonic() - stage_started) / 60:.1f} min", flush=True)
    completed = {
        **metadata,
        "elapsed_minutes": budget.elapsed_minutes,
        "training": {
            "validation_macro_f0_5": training["macro_f0_5"],
            "holdout_macro_f0_5": training["holdout_macro_f0_5"],
            "threshold": training["threshold"],
        },
        "inference": inference,
    }
    write_json(work_root / "completed_run.json", completed)
    return completed
