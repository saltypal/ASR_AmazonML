import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ber.config import load_config
from ber.pipeline import run_all


def _write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def _source_frame(prefix: str, count: int, split: str, role: str) -> pd.DataFrame:
    rows = []
    for index in range(count):
        if role == "positive" and split == "train" and index % 10 == 0:
            name = f"Unrelated Shop {index}"
            address = f"900 Far Road {50000 + index}"
        elif role == "positive":
            name = f"Cafe Number {index} Private Limited"
            address = f"{100 + index} Main Street {10000 + index}"
        elif role == "query":
            name = f"Cafe Number {index} Pvt Ltd"
            address = f"{100 + index}, Main St, {10000 + index}"
        else:
            name = f"Warehouse Different {index}"
            address = f"{800 + index} Other Avenue {70000 + index}"
        rows.append(
            {
                "entity_id": f"{prefix}-{index:05d}",
                "business_name": name,
                "business_address": address,
                "country": "US",
            }
        )
    return pd.DataFrame(rows)


class PipelineIntegrationTests(unittest.TestCase):
    def test_complete_cpu_pipeline_creates_validator_safe_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "dataset"
            for split, count in (("train", 120), ("test", 24)):
                _write_tsv(
                    data_root / split / f"{split}_source1.tsv",
                    _source_frame("S1", count, split, "query"),
                )
                _write_tsv(
                    data_root / split / f"{split}_source2.tsv",
                    _source_frame("S2", count, split, "positive"),
                )
                _write_tsv(
                    data_root / split / f"{split}_source3.tsv",
                    _source_frame("S3", count, split, "negative"),
                )
            truth = pd.DataFrame(
                {
                    "source1_entity_id": [f"S1-{index:05d}" for index in range(120)],
                    "matched_entity_ids": [
                        "" if index % 10 == 0 else f"S2-{index:05d}" for index in range(120)
                    ],
                }
            )
            _write_tsv(data_root / "train" / "train_ground_truth.tsv", truth)

            project_root = Path(__file__).resolve().parents[3]
            config = load_config(project_root / "code" / "business_entity_resolution" / "configs" / "smoke.yaml")
            config["project"].update(
                {
                    "sample_rows_per_file": None,
                    "time_budget_minutes": 10,
                    "output_reserve_minutes": 1,
                    "feature_chunk_rows": 75,
                }
            )
            config["candidate_generation"].update(
                {"name_top_k": 3, "address_top_k": 3, "max_candidates_per_query": 8}
            )
            config["training"].update(
                {
                    "tuning_trials": 1,
                    "tuning_timeout_seconds": 30,
                    "max_boost_rounds": 100,
                    "early_stopping_rounds": 10,
                }
            )
            completed = run_all(
                project_root,
                data_root,
                root / "work",
                root / "output",
                config,
            )
            matching = pd.read_csv(
                root / "output" / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False
            )
            candidates = pd.read_csv(
                root / "output" / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False
            )
            self.assertEqual(len(matching), 24)
            self.assertEqual(len(candidates), 24)
            self.assertGreater(int(matching["matched_entity_ids"].ne("").sum()), 0)
            self.assertIn("training", completed)
            self.assertGreaterEqual(completed["training"]["holdout_macro_f0_5"], 0.8)


if __name__ == "__main__":
    unittest.main()
