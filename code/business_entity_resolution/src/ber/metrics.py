from __future__ import annotations

import numpy as np
import pandas as pd


def entity_fbeta(expected: set[str], predicted: set[str], beta: float = 0.5) -> float:
    if not expected:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    true_positive = len(expected & predicted)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(predicted)
    recall = true_positive / len(expected)
    beta_squared = beta * beta
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def macro_fbeta(
    truth: dict[str, set[str]], predictions: dict[str, set[str]], beta: float = 0.5
) -> float:
    if not truth:
        raise ValueError("Cannot score an empty truth mapping.")
    scores = [entity_fbeta(expected, predictions.get(source1_id, set()), beta) for source1_id, expected in truth.items()]
    return float(np.mean(scores))


def predictions_at_threshold(scored_pairs: pd.DataFrame, threshold: float) -> dict[str, set[str]]:
    selected = scored_pairs.loc[
        scored_pairs["score"] >= threshold, ["source1_entity_id", "candidate_entity_id"]
    ]
    if selected.empty:
        return {}
    return selected.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()


def optimize_threshold(
    scored_pairs: pd.DataFrame,
    truth: dict[str, set[str]],
    minimum: float,
    maximum: float,
    steps: int,
) -> tuple[float, float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(minimum, maximum, steps):
        predictions = predictions_at_threshold(scored_pairs, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "macro_f0_5": macro_fbeta(truth, predictions, beta=0.5),
                "predicted_links": sum(len(values) for values in predictions.values()),
            }
        )
    curve = pd.DataFrame(rows)
    best = curve.sort_values(["macro_f0_5", "threshold"], ascending=[False, False]).iloc[0]
    return float(best.threshold), float(best.macro_f0_5), curve
