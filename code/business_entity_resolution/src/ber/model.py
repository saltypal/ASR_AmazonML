from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import FEATURE_COLUMNS
from .metrics import optimize_threshold


BASE_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "aucpr"],
    "tree_method": "hist",
    "max_depth": 8,
    "min_child_weight": 4,
    "eta": 0.08,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 3.0,
    "max_bin": 256,
    "seed": 42,
}


class _DeadlineCallback(xgb.callback.TrainingCallback):
    def __init__(self, deadline: float):
        self.deadline = deadline

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: dict) -> bool:
        return time.monotonic() >= self.deadline


def _matrix(
    frame: pd.DataFrame,
    labeled: bool,
    reference: xgb.QuantileDMatrix | None = None,
) -> xgb.QuantileDMatrix:
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
    label = frame["label"].to_numpy(dtype=np.float32, copy=False) if labeled else None
    return xgb.QuantileDMatrix(
        features, label=label, feature_names=FEATURE_COLUMNS, ref=reference
    )


def fit_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    params: dict[str, Any],
    max_rounds: int,
    early_stopping_rounds: int,
    deadline: float | None = None,
) -> tuple[xgb.Booster, dict[str, Any]]:
    train_matrix = _matrix(train_frame, labeled=True)
    validation_matrix = _matrix(validation_frame, labeled=True, reference=train_matrix)
    evaluations: dict[str, Any] = {}
    callbacks = [_DeadlineCallback(deadline)] if deadline is not None else None
    model = xgb.train(
        params,
        train_matrix,
        num_boost_round=max_rounds,
        evals=[(train_matrix, "train"), (validation_matrix, "validation")],
        evals_result=evaluations,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=callbacks,
        verbose_eval=False,
    )
    return model, evaluations


def score_frame(model: xgb.Booster, frame: pd.DataFrame) -> pd.DataFrame:
    matrix = xgb.DMatrix(
        frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False),
        feature_names=FEATURE_COLUMNS,
    )
    scored = frame[["source1_entity_id", "candidate_entity_id", "target_source"]].copy()
    best_iteration = getattr(model, "best_iteration", None)
    iteration_range = (0, int(best_iteration) + 1) if best_iteration is not None else (0, 0)
    scored["score"] = model.predict(matrix, iteration_range=iteration_range)
    return scored


def _sample_tuning_frame(frame: pd.DataFrame, maximum_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum_rows:
        return frame
    query_ids = frame["source1_entity_id"].drop_duplicates()
    mean_group_size = max(1.0, len(frame) / len(query_ids))
    number_of_queries = max(1, min(len(query_ids), int(maximum_rows / mean_group_size)))
    sampled_ids = set(query_ids.sample(n=number_of_queries, random_state=seed))
    sampled = frame[frame["source1_entity_id"].isin(sampled_ids)]
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def _parameter_candidates(seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    candidates = [dict(BASE_PARAMS)]
    for _ in range(64):
        params = dict(BASE_PARAMS)
        params.update(
            {
                "max_depth": int(rng.integers(5, 11)),
                "min_child_weight": float(np.exp(rng.uniform(np.log(1), np.log(20)))),
                "eta": float(np.exp(rng.uniform(np.log(0.025), np.log(0.16)))),
                "subsample": float(rng.uniform(0.68, 1.0)),
                "colsample_bytree": float(rng.uniform(0.65, 1.0)),
                "reg_alpha": float(np.exp(rng.uniform(np.log(1e-4), np.log(2.0)))),
                "reg_lambda": float(np.exp(rng.uniform(np.log(0.2), np.log(20.0)))),
            }
        )
        candidates.append(params)
    return candidates


def tune_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    validation_truth: dict[str, set[str]],
    config: dict[str, Any],
    seed: int,
) -> tuple[xgb.Booster, dict[str, Any]]:
    sample_limit = int(config["tuning_sample_rows"])
    tuning_train = _sample_tuning_frame(train_frame, sample_limit, seed)
    tuning_validation = _sample_tuning_frame(validation_frame, max(250_000, sample_limit // 4), seed + 1)
    tuning_validation_ids = set(tuning_validation["source1_entity_id"])
    tuning_truth = {
        source1_id: expected
        for source1_id, expected in validation_truth.items()
        if source1_id in tuning_validation_ids
    }
    timeout = float(config["tuning_timeout_seconds"])
    trial_limit = int(config["tuning_trials"])
    started = time.monotonic()
    search_deadline = started + timeout
    best_model: xgb.Booster | None = None
    best_result: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    for trial_number, params in enumerate(_parameter_candidates(seed)):
        if trial_number >= trial_limit or time.monotonic() - started >= timeout:
            break
        params = dict(params)
        params["seed"] = seed + trial_number
        params["device"] = str(config.get("device", "cpu"))
        model, history = fit_model(
            tuning_train,
            tuning_validation,
            params,
            int(config["max_boost_rounds"]),
            int(config["early_stopping_rounds"]),
            deadline=search_deadline,
        )
        scored = score_frame(model, tuning_validation)
        threshold, score, _ = optimize_threshold(
            scored,
            tuning_truth,
            float(config["threshold_min"]),
            float(config["threshold_max"]),
            int(config["threshold_steps"]),
        )
        result = {
            "trial": trial_number,
            "macro_f0_5": score,
            "threshold": threshold,
            "best_iteration": int(model.best_iteration),
            "params": params,
        }
        trials.append(result)
        if best_result is None or score > best_result["macro_f0_5"]:
            best_model, best_result = model, result
    if best_model is None or best_result is None:
        raise RuntimeError("No tuning trial completed inside the configured time budget.")
    search_elapsed = time.monotonic() - started
    final_params = dict(best_result["params"])
    final_model, history = fit_model(
        train_frame,
        validation_frame,
        final_params,
        int(config["max_boost_rounds"]),
        int(config["early_stopping_rounds"]),
    )
    validation_scores = score_frame(final_model, validation_frame)
    threshold, validation_score, threshold_curve = optimize_threshold(
        validation_scores,
        validation_truth,
        float(config["threshold_min"]),
        float(config["threshold_max"]),
        int(config["threshold_steps"]),
    )
    best_result = {
        **best_result,
        "threshold": threshold,
        "macro_f0_5": validation_score,
        "search_trials": trials,
        "search_elapsed_seconds": search_elapsed,
        "final_best_iteration": int(final_model.best_iteration),
        "final_history": history,
        "threshold_curve": threshold_curve.to_dict(orient="records"),
        "tuning_train_rows": len(tuning_train),
        "tuning_validation_rows": len(tuning_validation),
        "full_train_rows": len(train_frame),
        "full_validation_rows": len(validation_frame),
    }
    return final_model, best_result


def save_model_bundle(
    model: xgb.Booster,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "xgboost_model.json")
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_model_bundle(model_dir: Path) -> tuple[xgb.Booster, dict[str, Any]]:
    model = xgb.Booster()
    model.load_model(model_dir / "xgboost_model.json")
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    return model, metadata
