from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def build_embedding_text(frame: pd.DataFrame, prefix: str) -> list[str]:
    names = frame["name_norm"].fillna("").astype(str)
    addresses = frame["address_norm"].fillna("").astype(str)
    countries = frame["country"].fillna("").astype(str)
    return [
        f"{prefix}: business name: {name} | address: {address} | country: {country}"
        for name, address, country in zip(names, addresses, countries)
    ]


def _write_embedding_part(path: Path, entity_ids: list[str], embeddings: np.ndarray) -> None:
    embeddings = np.asarray(embeddings, dtype=np.float16)
    width = embeddings.shape[1]
    flat = pa.array(embeddings.reshape(-1), type=pa.float16())
    vectors = pa.FixedSizeListArray.from_arrays(flat, width)
    table = pa.table({"entity_id": pa.array(entity_ids), "embedding": vectors})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def encode_partition(
    input_path: Path,
    output_path: Path,
    model_name: str,
    prefix: str,
    batch_size: int,
    max_length: int,
    non_latin_only: bool,
) -> int:
    import torch
    from sentence_transformers import SentenceTransformer

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_length

    dataset = pq.ParquetDataset(input_path)
    rows_written = 0
    output_path.mkdir(parents=True, exist_ok=True)
    for batch_index, record_batch in enumerate(dataset.read(columns=[
        "entity_id", "name_norm", "address_norm", "country", "has_non_latin_name"
    ]).to_batches(max_chunksize=50_000)):
        if batch_index % world_size != rank:
            continue
        frame = record_batch.to_pandas()
        if non_latin_only:
            frame = frame[frame["has_non_latin_name"].astype(bool)]
        if frame.empty:
            continue
        texts = build_embedding_text(frame, prefix)
        with torch.inference_mode():
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                precision="float32",
            )
        part_path = output_path / f"rank-{rank:02d}-batch-{batch_index:06d}.parquet"
        _write_embedding_part(part_path, frame["entity_id"].tolist(), embeddings)
        rows_written += len(frame)
    (output_path / f"_SUCCESS.rank-{rank:02d}").write_text(str(rows_written), encoding="utf-8")
    return rows_written


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode one country/source partition on one or more GPUs.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix", choices=("query", "passage"), required=True)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--non-latin-only", action="store_true")
    args = parser.parse_args()
    rows = encode_partition(
        args.input,
        args.output,
        args.model,
        args.prefix,
        args.batch_size,
        args.max_length,
        args.non_latin_only,
    )
    print(f"Encoded {rows:,} records on local rank {os.environ.get('LOCAL_RANK', '0')}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
