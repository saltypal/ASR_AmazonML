from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


SIGNAL_COLUMNS = [
    "name_tfidf",
    "address_tfidf",
    "dense_score",
    "exact_name",
    "exact_address",
    "exact_name_core",
    "exact_numbers",
]


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "target_source", *SIGNAL_COLUMNS, "retrieval_score"]
    )


def _lane_frame(
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    target_source: str,
    signal: str,
    scores: np.ndarray | float,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "source1_entity_id": left_ids,
            "candidate_entity_id": right_ids,
            "target_source": target_source,
            signal: scores,
        }
    )
    return frame


def exact_block(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    key: str,
    target_source: str,
    signal: str,
    max_block_size: int,
) -> pd.DataFrame:
    right = candidates.loc[candidates[key].ne(""), ["entity_id", key]].copy()
    if right.empty:
        return _empty_candidates()
    frequencies = right[key].value_counts()
    allowed = frequencies[frequencies <= max_block_size].index
    right = right[right[key].isin(allowed)]
    left = queries.loc[queries[key].ne(""), ["entity_id", key]]
    joined = left.merge(right, on=key, how="inner", suffixes=("_left", "_right"), sort=False)
    if joined.empty:
        return _empty_candidates()
    return _lane_frame(
        joined["entity_id_left"].to_numpy(),
        joined["entity_id_right"].to_numpy(),
        target_source,
        signal,
        1.0,
    )


def _fallback_sparse_topn(
    left: sparse.csr_matrix,
    right_transposed: sparse.csc_matrix,
    top_k: int,
    threshold: float,
    chunk_size: int = 2_000,
) -> sparse.csr_matrix:
    blocks: list[sparse.csr_matrix] = []
    for start in range(0, left.shape[0], chunk_size):
        scores = (left[start : start + chunk_size] @ right_transposed).tocsr()
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for local_row in range(scores.shape[0]):
            begin, end = scores.indptr[local_row], scores.indptr[local_row + 1]
            row_values = scores.data[begin:end]
            row_columns = scores.indices[begin:end]
            keep_mask = row_values >= threshold
            row_values, row_columns = row_values[keep_mask], row_columns[keep_mask]
            if row_values.size > top_k:
                selected = np.argpartition(row_values, -top_k)[-top_k:]
                row_values, row_columns = row_values[selected], row_columns[selected]
            order = np.argsort(-row_values)
            rows.extend([local_row] * len(order))
            columns.extend(row_columns[order].tolist())
            values.extend(row_values[order].astype(float).tolist())
        blocks.append(
            sparse.csr_matrix(
                (values, (rows, columns)), shape=(scores.shape[0], right_transposed.shape[1])
            )
        )
    return sparse.vstack(blocks, format="csr") if blocks else sparse.csr_matrix((0, right_transposed.shape[1]))


def sparse_topn(
    left: sparse.csr_matrix,
    right: sparse.csr_matrix,
    top_k: int,
    threshold: float,
) -> sparse.csr_matrix:
    try:
        from sparse_dot_topn import sp_matmul_topn

        return sp_matmul_topn(
            left,
            right.T.tocsc(),
            top_n=top_k,
            threshold=threshold,
            sort=True,
        ).tocsr()
    except ImportError:
        return _fallback_sparse_topn(left, right.T.tocsc(), top_k, threshold)


def tfidf_block(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    text_column: str,
    target_source: str,
    signal: str,
    top_k: int,
    min_similarity: float,
    ngram_range: tuple[int, int],
    max_features: int,
) -> pd.DataFrame:
    if top_k <= 0 or queries.empty or candidates.empty:
        return _empty_candidates()
    query_text = queries[text_column].fillna("").astype(str)
    candidate_text = candidates[text_column].fillna("").astype(str)
    nonempty_query = query_text.ne("")
    nonempty_candidate = candidate_text.ne("")
    if not nonempty_query.any() or not nonempty_candidate.any():
        return _empty_candidates()

    query_subset = query_text[nonempty_query]
    candidate_subset = candidate_text[nonempty_candidate]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        dtype=np.float32,
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    combined = pd.concat([query_subset, candidate_subset], ignore_index=True)
    try:
        vectorizer.fit(combined)
    except ValueError:
        return _empty_candidates()
    left = vectorizer.transform(query_subset).tocsr()
    right = vectorizer.transform(candidate_subset).tocsr()
    similarities = sparse_topn(left, right, top_k=top_k, threshold=min_similarity)

    row_counts = np.diff(similarities.indptr)
    if row_counts.sum() == 0:
        return _empty_candidates()
    row_positions = np.repeat(np.arange(similarities.shape[0]), row_counts)
    query_ids = queries.loc[nonempty_query, "entity_id"].to_numpy()[row_positions]
    candidate_ids = candidates.loc[nonempty_candidate, "entity_id"].to_numpy()[similarities.indices]
    return _lane_frame(
        query_ids,
        candidate_ids,
        target_source,
        signal,
        similarities.data.astype(np.float32),
    )


def combine_candidate_lanes(
    lanes: Iterable[pd.DataFrame], max_candidates_per_query: int
) -> pd.DataFrame:
    usable = [lane for lane in lanes if lane is not None and not lane.empty]
    if not usable:
        return _empty_candidates()
    combined = pd.concat(usable, ignore_index=True, sort=False)
    for column in SIGNAL_COLUMNS:
        if column not in combined:
            combined[column] = 0.0
        combined[column] = combined[column].fillna(0).astype(np.float32)
    keys = ["source1_entity_id", "candidate_entity_id", "target_source"]
    combined = combined.groupby(keys, as_index=False, sort=False)[SIGNAL_COLUMNS].max()
    combined["retrieval_score"] = combined[["name_tfidf", "address_tfidf", "dense_score"]].max(axis=1)
    exact_columns = ["exact_name", "exact_address", "exact_name_core", "exact_numbers"]
    combined["retrieval_score"] += 0.35 * combined[exact_columns].max(axis=1)
    combined.sort_values(
        ["source1_entity_id", "retrieval_score", "candidate_entity_id"],
        ascending=[True, False, True],
        inplace=True,
        kind="stable",
    )
    combined = combined.groupby("source1_entity_id", sort=False, as_index=False).head(max_candidates_per_query)
    return combined.reset_index(drop=True)


def generate_candidates_for_source(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    target_source: str,
    config: dict,
) -> pd.DataFrame:
    ngram_range = (int(config["char_ngram_min"]), int(config["char_ngram_max"]))
    exact_specs = [
        ("name_norm", "exact_name"),
        ("address_punct", "exact_address"),
        ("name_core", "exact_name_core"),
        ("address_numbers", "exact_numbers"),
    ]
    lanes = [
        exact_block(
            queries,
            candidates,
            key,
            target_source,
            signal,
            int(config["max_exact_block_size"]),
        )
        for key, signal in exact_specs
    ]
    lanes.extend(
        [
            tfidf_block(
                queries,
                candidates,
                "name_punct",
                target_source,
                "name_tfidf",
                int(config["name_top_k"]),
                float(config["min_name_similarity"]),
                ngram_range,
                int(config["max_tfidf_features"]),
            ),
            tfidf_block(
                queries,
                candidates,
                "address_punct",
                target_source,
                "address_tfidf",
                int(config["address_top_k"]),
                float(config["min_address_similarity"]),
                ngram_range,
                int(config["max_tfidf_features"]),
            ),
        ]
    )
    return combine_candidate_lanes(lanes, int(config["max_candidates_per_query"]))


def candidate_recall(
    candidate_pairs: pd.DataFrame,
    truth: dict[str, set[str]],
) -> dict[str, float | int]:
    predicted = (
        candidate_pairs.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()
        if not candidate_pairs.empty
        else {}
    )
    total_links = recovered_links = entities_with_truth = entities_all_recovered = 0
    for source1_id, expected in truth.items():
        if not expected:
            continue
        entities_with_truth += 1
        total_links += len(expected)
        available = predicted.get(source1_id, set())
        recovered_links += len(expected & available)
        entities_all_recovered += int(expected <= available)
    return {
        "true_links": total_links,
        "recovered_true_links": recovered_links,
        "pair_recall": recovered_links / total_links if total_links else 1.0,
        "entities_with_truth": entities_with_truth,
        "entities_all_truth_recovered": entities_all_recovered,
        "all_matches_entity_recall": (
            entities_all_recovered / entities_with_truth if entities_with_truth else 1.0
        ),
        "candidate_pairs": len(candidate_pairs),
        "mean_candidates_per_query": (
            float(candidate_pairs.groupby("source1_entity_id").size().mean())
            if not candidate_pairs.empty
            else 0.0
        ),
    }
