from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer


SIGNAL_COLUMNS = [
    "name_tfidf",
    "address_tfidf",
    "token_tfidf",
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
    # Repeated values on either side can create a huge many-to-many join.
    left_frequencies = left[key].value_counts()
    left = left[left[key].isin(left_frequencies[left_frequencies <= max_block_size].index)]
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
    query_chunk_size: int = 2_000,
    deadline: float | None = None,
    max_document_frequency: int = 5_000,
) -> pd.DataFrame:
    if top_k <= 0 or queries.empty or candidates.empty:
        return _empty_candidates()
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(f"Candidate retrieval deadline reached before {target_source}/{signal}.")
    query_text = queries[text_column].fillna("").astype(str)
    candidate_text = candidates[text_column].fillna("").astype(str)
    nonempty_query = query_text.ne("")
    nonempty_candidate = candidate_text.ne("")
    if not nonempty_query.any() or not nonempty_candidate.any():
        return _empty_candidates()

    query_subset = query_text[nonempty_query]
    candidate_subset = candidate_text[nonempty_candidate]
    started = time.monotonic()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        dtype=np.float32,
        min_df=1,
        max_df=(
            max_document_frequency
            if len(candidate_subset) > max_document_frequency
            else 1.0
        ),
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    print(
        f"[candidates] {target_source}/{signal}: fitting TF-IDF and building index from "
        f"{len(candidate_subset):,} candidate texts (max_df={vectorizer.max_df})",
        flush=True,
    )
    try:
        right = vectorizer.fit_transform(candidate_subset).tocsr()
    except ValueError as exc:
        if "empty vocabulary" in str(exc) or "After pruning" in str(exc):
            print(f"[candidates] {target_source}/{signal}: no usable n-grams", flush=True)
            return _empty_candidates()
        raise
    print(
        f"[candidates] {target_source}/{signal}: index ready | "
        f"features={len(vectorizer.vocabulary_):,} | nnz={right.nnz:,} | "
        f"elapsed={(time.monotonic() - started) / 60:.1f} min",
        flush=True,
    )
    query_ids_all = queries.loc[nonempty_query, "entity_id"].to_numpy()
    candidate_ids_all = candidates.loc[nonempty_candidate, "entity_id"].to_numpy()
    blocks: list[pd.DataFrame] = []
    total_queries = len(query_subset)
    total_chunks = (total_queries + query_chunk_size - 1) // query_chunk_size
    for chunk_number, start in enumerate(range(0, total_queries, query_chunk_size), start=1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"Candidate retrieval deadline reached during {target_source}/{signal}."
            )
        end = min(start + query_chunk_size, total_queries)
        chunk_started = time.monotonic()
        print(
            f"[candidates] {target_source}/{signal}: transforming and searching chunk "
            f"{chunk_number}/{total_chunks} ({start:,}:{end:,} queries)",
            flush=True,
        )
        left = vectorizer.transform(query_subset.iloc[start:end]).tocsr()
        similarities = sparse_topn(
            left, right, top_k=top_k, threshold=min_similarity
        )
        row_counts = np.diff(similarities.indptr)
        if row_counts.sum():
            row_positions = np.repeat(np.arange(end - start), row_counts)
            blocks.append(
                _lane_frame(
                    query_ids_all[start:end][row_positions],
                    candidate_ids_all[similarities.indices],
                    target_source,
                    signal,
                    similarities.data.astype(np.float32),
                )
            )
        print(
            f"[candidates] {target_source}/{signal}: chunk {chunk_number}/{total_chunks} "
            f"complete | pairs={similarities.nnz:,} | "
            f"chunk_seconds={time.monotonic() - chunk_started:.1f} | "
            f"elapsed_minutes={(time.monotonic() - started) / 60:.1f}",
            flush=True,
        )
    return pd.concat(blocks, ignore_index=True) if blocks else _empty_candidates()


def _token_document(name: str, address: str, number_signature: str) -> str:
    """Keep name, address, and number evidence in separate hash namespaces."""
    terms = {f"n:{word}" for word in name.split() if len(word) >= 2}
    terms.update(f"a:{word}" for word in address.split() if len(word) >= 4 and not word.isdigit())
    terms.update(f"d:{number}" for number in number_signature.split("|") if len(number) >= 2)
    return " ".join(sorted(terms))


def _token_documents(frame: pd.DataFrame) -> list[str]:
    return [
        _token_document(name, address, numbers)
        for name, address, numbers in zip(
            frame["name_punct"], frame["address_punct"], frame["address_numbers"]
        )
    ]


def _address_token_documents(frame: pd.DataFrame) -> list[str]:
    """Provide an address-only view when business names use different scripts."""
    return [
        _token_document("", address, numbers)
        for address, numbers in zip(frame["address_punct"], frame["address_numbers"])
    ]


def token_hash_block(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    target_source: str,
    top_k: int,
    min_similarity: float,
    hash_features: int,
    query_chunk_size: int,
    max_document_frequency: int,
    threads: int,
    deadline: float | None = None,
    *,
    address_only: bool = False,
) -> pd.DataFrame:
    """Search a bounded sparse token index, retaining rare multilingual terms."""
    signal = "address_tfidf" if address_only else "token_tfidf"
    documents = _address_token_documents if address_only else _token_documents
    if top_k <= 0 or queries.empty or candidates.empty:
        return _empty_candidates()
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(f"Candidate deadline reached before {target_source}/{signal}.")

    from sparse_dot_topn import sp_matmul_topn

    started = time.monotonic()
    print(
        f"[candidates] {target_source}/{signal}: hashing {len(candidates):,} target records "
        f"into {hash_features:,} features",
        flush=True,
    )
    vectorizer = HashingVectorizer(
        analyzer=str.split,
        n_features=hash_features,
        alternate_sign=False,
        norm=None,
        dtype=np.float32,
    )
    right = vectorizer.transform(documents(candidates)).tocsr()
    frequencies = np.diff(right.tocsc().indptr)
    common = frequencies > max_document_frequency
    if common.any():
        right.data[common[right.indices]] = 0
        right.eliminate_zeros()
    transformer = TfidfTransformer(norm="l2", sublinear_tf=True)
    right = transformer.fit_transform(right).tocsr()
    right_transposed = right.T.tocsr()
    del right
    print(
        f"[candidates] {target_source}/{signal}: index ready | "
        f"nnz={right_transposed.nnz:,} | common_buckets={int(common.sum()):,} | "
        f"seconds={time.monotonic() - started:.1f}",
        flush=True,
    )

    query_ids = queries["entity_id"].to_numpy()
    candidate_ids = candidates["entity_id"].to_numpy()
    blocks: list[pd.DataFrame] = []
    total = len(queries)
    total_chunks = (total + query_chunk_size - 1) // query_chunk_size
    for number, start in enumerate(range(0, total, query_chunk_size), start=1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Candidate deadline reached during {target_source}/{signal}.")
        end = min(start + query_chunk_size, total)
        chunk_started = time.monotonic()
        left = vectorizer.transform(documents(queries.iloc[start:end])).tocsr()
        if common.any():
            left.data[common[left.indices]] = 0
            left.eliminate_zeros()
        left = transformer.transform(left).tocsr()
        similarities = sp_matmul_topn(
            left,
            right_transposed,
            top_n=top_k,
            threshold=min_similarity,
            n_threads=threads,
            sort=True,
        ).tocsr()
        row_counts = np.diff(similarities.indptr)
        if similarities.nnz:
            row_positions = np.repeat(np.arange(end - start), row_counts)
            blocks.append(
                _lane_frame(
                    query_ids[start:end][row_positions],
                    candidate_ids[similarities.indices],
                    target_source,
                    signal,
                    similarities.data.astype(np.float32),
                )
            )
        print(
            f"[candidates] {target_source}/{signal}: chunk {number}/{total_chunks} | "
            f"queries={end:,}/{total:,} | pairs={similarities.nnz:,} | "
            f"chunk_seconds={time.monotonic() - chunk_started:.1f} | "
            f"elapsed_minutes={(time.monotonic() - started) / 60:.1f}",
            flush=True,
        )
    return pd.concat(blocks, ignore_index=True) if blocks else _empty_candidates()


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
    combined["retrieval_score"] = combined[
        ["name_tfidf", "address_tfidf", "token_tfidf", "dense_score"]
    ].max(axis=1)
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
    deadline: float | None = None,
) -> pd.DataFrame:
    ngram_range = (int(config["char_ngram_min"]), int(config["char_ngram_max"]))
    # Exact blocks are bounded on both sides by exact_limit, so repeated values
    # cannot create an unbounded many-to-many join.
    exact_specs = [
        ("name_norm", "exact_name"),
        ("address_punct", "exact_address"),
        ("name_core", "exact_name_core"),
        ("address_numbers", "exact_numbers"),
    ]
    exact_limit = min(
        int(config["max_exact_block_size"]),
        int(config["max_candidates_per_query"]),
    )
    lanes: list[pd.DataFrame] = []
    for key, signal in exact_specs:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Candidate retrieval deadline reached before {target_source}/{signal}.")
        lane_started = time.monotonic()
        print(f"[candidates] {target_source}/{signal}: exact blocking", flush=True)
        lane = exact_block(
            queries,
            candidates,
            key,
            target_source,
            signal,
            exact_limit,
        )
        lanes.append(lane)
        print(
            f"[candidates] {target_source}/{signal}: {len(lane):,} pairs | "
            f"elapsed_seconds={time.monotonic() - lane_started:.1f}",
            flush=True,
        )

    method = config.get("method", "char_tfidf")
    print(
        f"[candidates] {target_source}: {method} retrieval for all {len(queries):,} queries; "
        "one Source-1 entity may have multiple true matches",
        flush=True,
    )
    if method == "token_hash":
        lanes.append(
            token_hash_block(
                queries,
                candidates,
                target_source,
                int(config["token_top_k"]),
                float(config["token_min_similarity"]),
                int(config["token_hash_features"]),
                int(config["token_query_chunk_size"]),
                int(config["max_tfidf_document_frequency"]),
                int(config["token_threads"]),
                deadline,
            )
        )
        if int(config.get("token_address_top_k", 0)) > 0:
            lanes.append(
                token_hash_block(
                    queries,
                    candidates,
                    target_source,
                    int(config["token_address_top_k"]),
                    float(config["token_min_similarity"]),
                    int(config["token_hash_features"]),
                    int(config["token_query_chunk_size"]),
                    int(config["token_address_max_document_frequency"]),
                    int(config["token_threads"]),
                    deadline,
                    address_only=True,
                )
            )
    elif method == "char_tfidf":
        lanes.extend([
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
                int(config.get("tfidf_query_chunk_size", 2_000)),
                deadline,
                int(config.get("max_tfidf_document_frequency", 5_000)),
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
                int(config.get("tfidf_query_chunk_size", 2_000)),
                deadline,
                int(config.get("max_tfidf_document_frequency", 5_000)),
            ),
        ])
    else:
        raise ValueError(f"Unsupported candidate method: {method}")
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
