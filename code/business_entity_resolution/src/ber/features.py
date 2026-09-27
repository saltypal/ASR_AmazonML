from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .candidates import SIGNAL_COLUMNS
from .normalization import TOKEN_PATTERN


FEATURE_COLUMNS = [
    "name_retrieval_lane",
    "address_retrieval_lane",
    "token_retrieval_lane",
    "dense_retrieval_lane",
    "retrieval_rank_score",
    "dense_score",
    "exact_name",
    "exact_address",
    "exact_name_core",
    "exact_numbers",
    "name_ratio",
    "name_token_set_ratio",
    "name_jaccard",
    "address_ratio",
    "address_token_set_ratio",
    "address_jaccard",
    "number_jaccard",
    "postal_exact",
    "name_exact_normalized",
    "name_core_exact",
    "address_exact_normalized",
    "country_equal",
    "name_script_equal",
    "cross_script_name",
    "query_address_missing",
    "candidate_address_missing",
    "name_length_ratio",
    "address_length_ratio",
    "target_is_source3",
]

FEATURE_WORKERS = min(4, os.cpu_count() or 1)


def _ratio(left: str, right: str) -> float:
    return fuzz.ratio(left, right, score_cutoff=0) / 100.0


def _token_set_ratio(left: str, right: str) -> float:
    return fuzz.token_set_ratio(left, right, score_cutoff=0) / 100.0


@lru_cache(maxsize=16_384)
def _normalized_tokens(text: str) -> frozenset[str]:
    # Each query participates in many candidate pairs, so parse its tokens once.
    return frozenset(TOKEN_PATTERN.findall(text))


def _normalized_token_jaccard(left: str, right: str) -> float:
    """The inputs are already Unicode and punctuation normalized at ingest."""
    left_tokens = _normalized_tokens(left)
    right_tokens = _normalized_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _number_signature_jaccard(left: str, right: str) -> float:
    """Compare the stored number signatures without re-normalizing addresses."""
    left_numbers = set(filter(None, left.split("|")))
    right_numbers = set(filter(None, right.split("|")))
    if not left_numbers and not right_numbers:
        return 0.0
    union = left_numbers | right_numbers
    return len(left_numbers & right_numbers) / len(union) if union else 0.0


def _length_ratio(left: pd.Series, right: pd.Series) -> np.ndarray:
    left_length = left.str.len().to_numpy(dtype=np.float32)
    right_length = right.str.len().to_numpy(dtype=np.float32)
    maximum = np.maximum(left_length, right_length)
    minimum = np.minimum(left_length, right_length)
    return np.divide(minimum, maximum, out=np.ones_like(minimum), where=maximum > 0)


def build_pair_features(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "target_source", *FEATURE_COLUMNS])
    query_columns = [
        "entity_id", "name_norm", "name_punct", "name_core", "address_norm", "address_punct",
        "address_numbers", "postal_token", "name_script", "country",
    ]
    candidate_columns = query_columns
    query_lookup = queries[query_columns].rename(
        columns={column: f"q_{column}" for column in query_columns if column != "entity_id"}
    ).rename(columns={"entity_id": "source1_entity_id"})
    candidate_lookup = candidates[candidate_columns].rename(
        columns={column: f"c_{column}" for column in candidate_columns if column != "entity_id"}
    ).rename(columns={"entity_id": "candidate_entity_id"})
    merged = pairs.merge(query_lookup, on="source1_entity_id", how="left", validate="many_to_one")
    merged = merged.merge(candidate_lookup, on="candidate_entity_id", how="left", validate="many_to_one")
    required_text = ["q_name_punct", "c_name_punct", "q_address_punct", "c_address_punct"]
    if merged[required_text].isna().any().any():
        raise ValueError("Candidate pairs contain IDs missing from the supplied source partitions.")
    for column in SIGNAL_COLUMNS + ["retrieval_score"]:
        if column not in merged:
            merged[column] = 0.0
        merged[column] = merged[column].fillna(0).astype(np.float32)
    merged["name_retrieval_lane"] = merged["name_tfidf"].gt(0).astype(np.float32)
    merged["address_retrieval_lane"] = merged["address_tfidf"].gt(0).astype(np.float32)
    merged["token_retrieval_lane"] = merged["token_tfidf"].gt(0).astype(np.float32)
    merged["dense_retrieval_lane"] = merged["dense_score"].gt(0).astype(np.float32)
    retrieval_rank = merged.groupby("source1_entity_id")["retrieval_score"].rank(
        method="first", ascending=False
    )
    group_size = merged.groupby("source1_entity_id")["retrieval_score"].transform("size")
    denominator = (group_size - 1).clip(lower=1)
    merged["retrieval_rank_score"] = (1.0 - (retrieval_rank - 1.0) / denominator).astype(np.float32)

    q_name = merged["q_name_punct"].astype(str)
    c_name = merged["c_name_punct"].astype(str)
    q_address = merged["q_address_punct"].astype(str)
    c_address = merged["c_address_punct"].astype(str)
    q_name_list = q_name.tolist()
    c_name_list = c_name.tolist()
    q_address_list = q_address.tolist()
    c_address_list = c_address.tolist()
    merged["name_ratio"] = process.cpdist(
        q_name_list, c_name_list, scorer=fuzz.ratio,
        workers=FEATURE_WORKERS, dtype=np.float32,
    ) / 100.0
    merged["address_ratio"] = process.cpdist(
        q_address_list, c_address_list, scorer=fuzz.ratio,
        workers=FEATURE_WORKERS, dtype=np.float32,
    ) / 100.0
    merged["name_token_set_ratio"] = process.cpdist(
        q_name_list, c_name_list, scorer=fuzz.token_set_ratio,
        workers=FEATURE_WORKERS, dtype=np.float32,
    ) / 100.0
    merged["address_token_set_ratio"] = process.cpdist(
        q_address_list, c_address_list, scorer=fuzz.token_set_ratio,
        workers=FEATURE_WORKERS, dtype=np.float32,
    ) / 100.0
    merged["name_jaccard"] = np.fromiter(
        (_normalized_token_jaccard(a, b) for a, b in zip(q_name, c_name)), dtype=np.float32
    )
    merged["address_jaccard"] = np.fromiter(
        (_normalized_token_jaccard(a, b) for a, b in zip(q_address, c_address)), dtype=np.float32
    )
    merged["number_jaccard"] = np.fromiter(
        (_number_signature_jaccard(a, b) for a, b in zip(merged["q_address_numbers"], merged["c_address_numbers"])),
        dtype=np.float32,
    )
    merged["postal_exact"] = (
        merged["q_postal_token"].ne("") & merged["q_postal_token"].eq(merged["c_postal_token"])
    ).astype(np.float32)
    merged["name_exact_normalized"] = merged["q_name_norm"].eq(merged["c_name_norm"]).astype(np.float32)
    merged["name_core_exact"] = (
        merged["q_name_core"].ne("") & merged["q_name_core"].eq(merged["c_name_core"])
    ).astype(np.float32)
    merged["address_exact_normalized"] = (
        merged["q_address_norm"].ne("") & merged["q_address_norm"].eq(merged["c_address_norm"])
    ).astype(np.float32)
    merged["country_equal"] = merged["q_country"].eq(merged["c_country"]).astype(np.float32)
    merged["name_script_equal"] = merged["q_name_script"].eq(merged["c_name_script"]).astype(np.float32)
    merged["cross_script_name"] = (
        merged["q_name_script"].ne(merged["c_name_script"])
        & merged["q_name_script"].ne("NONE")
        & merged["c_name_script"].ne("NONE")
    ).astype(np.float32)
    merged["query_address_missing"] = merged["q_address_norm"].eq("").astype(np.float32)
    merged["candidate_address_missing"] = merged["c_address_norm"].eq("").astype(np.float32)
    merged["name_length_ratio"] = _length_ratio(q_name, c_name)
    merged["address_length_ratio"] = _length_ratio(q_address, c_address)
    merged["target_is_source3"] = merged["target_source"].eq("S3").astype(np.float32)

    output_columns = ["source1_entity_id", "candidate_entity_id", "target_source", *FEATURE_COLUMNS]
    return merged[output_columns].copy()


def label_pair_features(features: pd.DataFrame, truth: dict[str, set[str]]) -> pd.DataFrame:
    labeled = features.copy()
    labeled["label"] = np.fromiter(
        (
            candidate_id in truth.get(source1_id, set())
            for source1_id, candidate_id in zip(
                labeled["source1_entity_id"], labeled["candidate_entity_id"]
            )
        ),
        dtype=np.int8,
    )
    return labeled
