from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .normalization import prepare_source_frame
from .runtime import write_json


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
SOURCE_NAMES = ("source1", "source2", "source3")
NORMALIZATION_VERSION = "nfkc-casefold-v1"


def dataset_files(data_root: Path, split: str) -> dict[str, Path]:
    root = data_root / split
    paths = {source: root / f"{split}_{source}.tsv" for source in SOURCE_NAMES}
    if split == "train":
        paths["ground_truth"] = root / "train_ground_truth.tsv"
    return paths


def validate_input_headers(data_root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for split in ("train", "test"):
        for name, path in dataset_files(data_root, split).items():
            if not path.is_file():
                raise FileNotFoundError(f"Required challenge file not found: {path}")
            expected = GROUND_TRUTH_COLUMNS if name == "ground_truth" else SOURCE_COLUMNS
            columns = pd.read_csv(path, sep="\t", nrows=0, encoding="utf-8").columns.tolist()
            if columns != expected:
                raise ValueError(f"{path.name} columns are {columns}; expected {expected}.")
            inventory[f"{split}_{name}"] = {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "columns": columns,
            }
    return inventory


def iter_tsv(path: Path, chunksize: int, nrows: int | None = None) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
        nrows=nrows,
        on_bad_lines="error",
    )


def _country_slug(country: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", country.casefold()).strip("-")
    return slug or "empty"


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def preprocess_source(
    path: Path,
    destination: Path,
    expected_prefix: str,
    chunksize: int,
    nrows: int | None,
) -> dict[str, Any]:
    file_size_gib = path.stat().st_size / (1024**3)
    print(
        f"[preprocess] START {path.name} | size={file_size_gib:.2f} GiB | "
        f"chunk_size={chunksize:,} rows",
        flush=True,
    )
    destination.mkdir(parents=True, exist_ok=True)
    row_count = 0
    empty_addresses = 0
    countries: Counter[str] = Counter()
    part_counts: Counter[str] = Counter()
    id_hashes: list[np.ndarray] = []

    for chunk_number, chunk in enumerate(
        iter_tsv(path, chunksize=chunksize, nrows=nrows), start=1
    ):
        if chunk.columns.tolist() != SOURCE_COLUMNS:
            raise ValueError(f"Unexpected schema while reading {path}: {chunk.columns.tolist()}")
        if not chunk["entity_id"].str.startswith(expected_prefix).all():
            bad = chunk.loc[~chunk["entity_id"].str.startswith(expected_prefix), "entity_id"].head().tolist()
            raise ValueError(f"{path.name} has IDs with the wrong prefix: {bad}")
        if chunk["entity_id"].eq("").any():
            raise ValueError(f"{path.name} contains an empty entity_id.")
        id_hashes.append(pd.util.hash_array(chunk["entity_id"].to_numpy(), encoding="utf8"))
        prepared = prepare_source_frame(chunk)
        row_count += len(prepared)
        empty_addresses += int(prepared["business_address"].eq("").sum())

        for country, country_frame in prepared.groupby("country", sort=False, dropna=False):
            country = str(country)
            countries[country] += len(country_frame)
            slug = _country_slug(country)
            part_number = part_counts[slug]
            part_counts[slug] += 1
            part_path = destination / f"country={slug}" / f"part-{part_number:05d}.parquet"
            _atomic_parquet(country_frame, part_path)

        if chunk_number == 1 or chunk_number % 5 == 0:
            print(
                f"[preprocess] {path.name}: chunk={chunk_number:,} | "
                f"rows_processed={row_count:,} | countries_seen={len(countries):,}",
                flush=True,
            )

    if id_hashes:
        all_hashes = np.concatenate(id_hashes)
        if np.unique(all_hashes).size != all_hashes.size:
            raise ValueError(f"{path.name} contains duplicate entity IDs in the scanned rows.")

    manifest = {
        "source_file": str(path.resolve()),
        "normalization_version": NORMALIZATION_VERSION,
        "rows": row_count,
        "empty_addresses": empty_addresses,
        "countries": dict(sorted(countries.items())),
        "sampled": nrows is not None,
        "sample_rows_requested": nrows,
    }
    write_json(destination / "manifest.json", manifest)
    print(
        f"[preprocess] DONE  {path.name} | rows={row_count:,} | "
        f"countries={len(countries):,} | empty_addresses={empty_addresses:,}",
        flush=True,
    )
    return manifest


def preprocess_dataset(data_root: Path, output_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    inventory = validate_input_headers(data_root)
    chunksize = int(config["project"]["chunksize"])
    nrows = config["project"].get("sample_rows_per_file")
    summary: dict[str, Any] = {
        "normalization_version": NORMALIZATION_VERSION,
        "input_inventory": inventory,
        "sources": {},
    }
    for split in ("train", "test"):
        paths = dataset_files(data_root, split)
        for source_index, source in enumerate(SOURCE_NAMES, start=1):
            key = f"{split}_{source}"
            summary["sources"][key] = preprocess_source(
                paths[source],
                output_root / split / source,
                expected_prefix=f"S{source_index}-",
                chunksize=chunksize,
                nrows=nrows,
            )
    write_json(output_root / "preprocessing_manifest.json", summary)
    return summary


def load_country_partition(root: Path, split: str, source: str, country_slug: str) -> pd.DataFrame:
    paths = sorted((root / split / source / f"country={country_slug}").glob("part-*.parquet"))
    if not paths:
        return pd.DataFrame(columns=SOURCE_COLUMNS)
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def available_country_slugs(root: Path, split: str, source: str = "source1") -> list[str]:
    source_root = root / split / source
    return sorted(path.name.split("=", 1)[1] for path in source_root.glob("country=*") if path.is_dir())


def load_ground_truth(path: Path, nrows: int | None = None) -> pd.DataFrame:
    truth = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )
    if truth.columns.tolist() != GROUND_TRUTH_COLUMNS:
        raise ValueError(f"Unexpected ground-truth schema: {truth.columns.tolist()}")
    return truth


def ground_truth_sets(truth: pd.DataFrame) -> dict[str, set[str]]:
    return {
        row.source1_entity_id: set(filter(None, row.matched_entity_ids.split(",")))
        for row in truth.itertuples(index=False)
    }
