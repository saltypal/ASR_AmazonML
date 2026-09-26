import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ber.io import load_country_partition, preprocess_source


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


if __name__ == "__main__":
    unittest.main()
