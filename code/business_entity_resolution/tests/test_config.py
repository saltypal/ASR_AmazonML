from pathlib import Path
import unittest

from ber.config import load_config


class ConfigTests(unittest.TestCase):
    def test_smoke_config_has_bounded_runtime(self):
        config_path = Path(__file__).parents[1] / "configs" / "smoke.yaml"
        config = load_config(config_path)
        self.assertEqual(config["project"]["time_budget_minutes"], 30)
        self.assertFalse(config["embeddings"]["enabled"])
        self.assertEqual(len(config["_config_hash"]), 12)


if __name__ == "__main__":
    unittest.main()
