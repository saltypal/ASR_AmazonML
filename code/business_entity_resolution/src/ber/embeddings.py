from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def launch_parallel_encoding(
    input_path: Path,
    output_path: Path,
    model_name: str,
    prefix: str,
    batch_size_per_gpu: int,
    max_length: int,
    gpu_count: int,
    non_latin_only: bool = False,
    timeout_seconds: float | None = None,
) -> None:
    if gpu_count < 1:
        raise RuntimeError("Embedding encoding requires at least one CUDA GPU in production mode.")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(gpu_count),
        "-m",
        "ber.embedding_worker",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model",
        model_name,
        "--prefix",
        prefix,
        "--batch-size",
        str(batch_size_per_gpu),
        "--max-length",
        str(max_length),
    ]
    if non_latin_only:
        command.append("--non-latin-only")
    environment = os.environ.copy()
    process = subprocess.Popen(
        command,
        env=environment,
        start_new_session=os.name != "nt",
    )
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise TimeoutError("Distributed embedding encoding exceeded its time budget.") from error
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    success_files = list(output_path.glob("_SUCCESS.rank-*"))
    if len(success_files) != gpu_count:
        raise RuntimeError(
            f"Expected {gpu_count} embedding workers to finish; found {len(success_files)} success markers."
        )


def iter_embedding_parts(root: Path) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for path in sorted(root.glob("rank-*-batch-*.parquet")):
        table = pq.read_table(path)
        entity_ids = np.asarray(table["entity_id"].to_pylist(), dtype=str)
        vectors = np.asarray(table["embedding"].to_pylist(), dtype=np.float32)
        yield entity_ids, vectors


def load_embeddings(root: Path) -> tuple[np.ndarray, np.ndarray]:
    id_parts: list[np.ndarray] = []
    vector_parts: list[np.ndarray] = []
    for entity_ids, vectors in iter_embedding_parts(root):
        id_parts.append(entity_ids)
        vector_parts.append(vectors)
    if not id_parts:
        return np.asarray([], dtype=str), np.empty((0, 0), dtype=np.float32)
    return np.concatenate(id_parts), np.vstack(vector_parts)


def dense_candidates_from_arrays(
    query_ids: np.ndarray,
    query_vectors: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_vectors: np.ndarray,
    target_source: str,
    top_k: int,
    min_similarity: float = 0.0,
) -> pd.DataFrame:
    if not len(query_ids) or not len(candidate_ids) or top_k <= 0:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "target_source", "dense_score"]
        )
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    candidate_vectors = np.asarray(candidate_vectors, dtype=np.float32)
    query_vectors /= np.maximum(np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-12)
    candidate_vectors /= np.maximum(np.linalg.norm(candidate_vectors, axis=1, keepdims=True), 1e-12)
    requested = min(top_k, len(candidate_ids))
    try:
        import faiss

        index = faiss.IndexFlatIP(candidate_vectors.shape[1])
        index.add(candidate_vectors)
        scores, positions = index.search(query_vectors, requested)
    except ImportError:
        from sklearn.neighbors import NearestNeighbors

        neighbors = NearestNeighbors(n_neighbors=requested, metric="cosine", algorithm="brute")
        neighbors.fit(candidate_vectors)
        distances, positions = neighbors.kneighbors(query_vectors)
        scores = 1.0 - distances
    repeated_queries = np.repeat(query_ids, requested)
    flat_positions = positions.reshape(-1)
    flat_scores = scores.reshape(-1)
    keep = flat_scores >= min_similarity
    return pd.DataFrame(
        {
            "source1_entity_id": repeated_queries[keep],
            "candidate_entity_id": candidate_ids[flat_positions[keep]],
            "target_source": target_source,
            "dense_score": flat_scores[keep].astype(np.float32),
        }
    )


def generate_dense_candidates(
    query_embedding_root: Path,
    candidate_embedding_root: Path,
    target_source: str,
    top_k: int,
    min_similarity: float = 0.0,
) -> pd.DataFrame:
    query_ids, query_vectors = load_embeddings(query_embedding_root)
    candidate_ids, candidate_vectors = load_embeddings(candidate_embedding_root)
    return dense_candidates_from_arrays(
        query_ids,
        query_vectors,
        candidate_ids,
        candidate_vectors,
        target_source,
        top_k,
        min_similarity,
    )
