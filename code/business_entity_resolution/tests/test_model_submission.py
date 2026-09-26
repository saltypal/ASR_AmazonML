import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ber.features import FEATURE_COLUMNS
from ber.model import score_frame, tune_model
from ber.submission import combine_output_parts, validate_output_relationships, write_country_output_parts


def _model_frame(query_start: int, query_count: int) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    rows = []
    truth = {}
    for offset in range(query_count):
        query_number = query_start + offset
        query_id = f"S1-{query_number:05d}"
        positive_id = f"S2-{query_number:05d}"
        truth[query_id] = {positive_id}
        for is_positive in (1, 0):
            row = {
                "source1_entity_id": query_id,
                "candidate_entity_id": positive_id if is_positive else f"S3-{query_number + 5000:05d}",
                "target_source": "S2" if is_positive else "S3",
                "label": is_positive,
            }
            for feature in FEATURE_COLUMNS:
                row[feature] = np.float32(0.9 if is_positive else 0.1)
            rows.append(row)
    return pd.DataFrame(rows), truth


class ModelAndSubmissionTests(unittest.TestCase):
    def test_xgboost_tuning_and_threshold_integration(self):
        train, _ = _model_frame(0, 30)
        validation, truth = _model_frame(100, 12)
        config = {
            "tuning_sample_rows": 1000,
            "tuning_timeout_seconds": 30,
            "tuning_trials": 1,
            "max_boost_rounds": 20,
            "early_stopping_rounds": 4,
            "threshold_min": 0.2,
            "threshold_max": 0.9,
            "threshold_steps": 8,
            "device": "cpu",
        }
        model, metadata = tune_model(train, validation, truth, config, seed=7)
        scored = score_frame(model, validation)
        self.assertEqual(len(scored), len(validation))
        self.assertGreaterEqual(metadata["macro_f0_5"], 0.9)
        self.assertIn("threshold", metadata)

    def test_submission_contains_all_source1_rows_and_subset(self):
        source1_ids = pd.Series(["S1-2", "S1-1", "S1-3"])
        pairs = pd.DataFrame(
            {
                "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
                "candidate_entity_id": ["S2-1", "S3-1", "S2-2"],
            }
        )
        scored = pairs.copy()
        scored["target_source"] = ["S2", "S3", "S2"]
        scored["score"] = [0.95, 0.2, 0.91]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_country_output_parts(source1_ids, pairs, scored, 0.8, root, "us")
            matching, candidates = combine_output_parts(root)
            summary = validate_output_relationships(matching, candidates)
            matching_frame = pd.read_csv(matching, sep="\t", dtype=str, keep_default_na=False)
            self.assertEqual(len(matching_frame), 3)
            self.assertEqual(summary["matched_links"], 2)
            self.assertEqual(
                matching_frame.loc[
                    matching_frame["source1_entity_id"] == "S1-3", "matched_entity_ids"
                ].iloc[0],
                "",
            )


if __name__ == "__main__":
    unittest.main()
