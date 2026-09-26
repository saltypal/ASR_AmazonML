from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd


def _group_ids(frame: pd.DataFrame, value_column: str) -> dict[str, str]:
    if frame.empty:
        return {}
    grouped = (
        frame.groupby("source1_entity_id", sort=False)[value_column]
        .agg(lambda values: ",".join(sorted(set(map(str, values)))))
    )
    return grouped.to_dict()


def _write_mapping(
    path: Path,
    source1_ids: Iterable[str],
    mapping: dict[str, str],
    value_header: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", value_header])
        for source1_id in source1_ids:
            writer.writerow([source1_id, mapping.get(source1_id, "")])
    temporary.replace(path)


def write_country_output_parts(
    source1_ids: pd.Series,
    candidate_pairs: pd.DataFrame,
    scored_pairs: pd.DataFrame,
    threshold: float,
    output_root: Path,
    country_slug: str,
) -> tuple[Path, Path]:
    ordered_ids = sorted(source1_ids.astype(str).unique())
    candidate_mapping = _group_ids(candidate_pairs, "candidate_entity_id")
    selected = scored_pairs.loc[scored_pairs["score"] >= threshold]
    match_mapping = _group_ids(selected, "candidate_entity_id")

    parts_root = output_root / "parts"
    matching_path = parts_root / f"matching-{country_slug}.tsv"
    candidate_path = parts_root / f"candidate-{country_slug}.tsv"
    _write_mapping(matching_path, ordered_ids, match_mapping, "matched_entity_ids")
    _write_mapping(candidate_path, ordered_ids, candidate_mapping, "candidate_entity_ids")
    return matching_path, candidate_path


def combine_output_parts(output_root: Path) -> tuple[Path, Path]:
    specifications = [
        ("matching-*.tsv", "matching_results.tsv", "matched_entity_ids"),
        ("candidate-*.tsv", "candidate_pairs.tsv", "candidate_entity_ids"),
    ]
    results: list[Path] = []
    for pattern, filename, value_header in specifications:
        parts = sorted((output_root / "parts").glob(pattern))
        if not parts:
            raise FileNotFoundError(f"No output parts found for {pattern}.")
        frames = [pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False) for path in parts]
        combined = pd.concat(frames, ignore_index=True)
        expected = ["source1_entity_id", value_header]
        if combined.columns.tolist() != expected:
            raise ValueError(f"Unexpected output columns for {filename}: {combined.columns.tolist()}")
        if combined["source1_entity_id"].duplicated().any():
            raise ValueError(f"Duplicate Source-1 rows while combining {filename}.")
        combined.sort_values("source1_entity_id", inplace=True, kind="stable")
        destination = output_root / filename
        temporary = destination.with_suffix(destination.suffix + ".partial")
        combined.to_csv(temporary, sep="\t", index=False, lineterminator="\n")
        temporary.replace(destination)
        results.append(destination)
    return results[0], results[1]


def validate_output_relationships(matching_path: Path, candidate_path: Path) -> dict[str, int]:
    matching = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    candidates = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
    if set(matching["source1_entity_id"]) != set(candidates["source1_entity_id"]):
        raise ValueError("Matching and candidate files do not contain identical Source-1 IDs.")
    candidate_map = {
        row.source1_entity_id: set(filter(None, row.candidate_entity_ids.split(",")))
        for row in candidates.itertuples(index=False)
    }
    violations = []
    matched_links = candidate_links = 0
    for row in matching.itertuples(index=False):
        matched = set(filter(None, row.matched_entity_ids.split(",")))
        available = candidate_map[row.source1_entity_id]
        matched_links += len(matched)
        candidate_links += len(available)
        if not matched <= available:
            violations.append(row.source1_entity_id)
    if violations:
        raise ValueError(
            f"Final matches are not a subset of candidates for {len(violations)} Source-1 rows."
        )
    return {
        "source1_rows": len(matching),
        "matched_links": matched_links,
        "candidate_links": candidate_links,
    }
