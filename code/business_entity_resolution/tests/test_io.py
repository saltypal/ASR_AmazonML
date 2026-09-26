import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ber.io import load_country_partition, load_ground_truth, preprocess_source


class PreprocessingTests(unittest.TestCase):
    def test_preprocess_source_partitions_and_preserves_unicode(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "train_source2.tsv"
            frame = pd.DataFrame(
                [
                    ["S2-1", "Ａcme Ltd", "14 Main Rd 282005", "India"],
                    ["S2-2", "लक्ष्मी प्रा. लि.", "14 Main Road 282005", "India"],
                    ["S2-3", "Société Test", "1 Rue A", "France"],
                ],
                columns=["entity_id", "business_name", "business_address", "country"],
            )
            frame.to_csv(source_path, sep="\t", index=False)
            output = root / "prepared" / "train" / "source2"
            manifest = preprocess_source(source_path, output, "S2-", chunksize=2, nrows=None)
            self.assertEqual(manifest["rows"], 3)
            india = load_country_partition(root / "prepared", "train", "source2", "india")
            self.assertEqual(len(india), 2)
            self.assertIn("लक्ष्मी", india.loc[1, "name_norm"])
            self.assertEqual(india.loc[0, "name_norm"], "acme ltd")

    def test_ground_truth_rejects_duplicate_rows_and_invalid_targets(self):
        invalid_frames = [
            pd.DataFrame(
                [["S1-1", "S2-1"], ["S1-1", "S2-2"]],
                columns=["source1_entity_id", "matched_entity_ids"],
            ),
            pd.DataFrame(
                [["S1-1", "S1-9"]],
                columns=["source1_entity_id", "matched_entity_ids"],
            ),
            pd.DataFrame(
                [["S1-1", "S2-1,S2-1"]],
                columns=["source1_entity_id", "matched_entity_ids"],
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_ground_truth.tsv"
            for frame in invalid_frames:
                frame.to_csv(path, sep="\t", index=False)
                with self.assertRaises(ValueError):
                    load_ground_truth(path)

    def test_ground_truth_trims_id_whitespace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_ground_truth.tsv"
            pd.DataFrame(
                [[" S1-1 ", " S2-1, S3-2 "]],
                columns=["source1_entity_id", "matched_entity_ids"],
            ).to_csv(path, sep="\t", index=False)
            truth = load_ground_truth(path)
            self.assertEqual(truth.iloc[0].to_dict(), {
                "source1_entity_id": "S1-1",
                "matched_entity_ids": "S2-1,S3-2",
            })


if __name__ == "__main__":
    unittest.main()
