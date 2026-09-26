from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from ber.config import DEFAULT_CONFIG, load_config, validate_config


class ConfigTests(unittest.TestCase):
    def test_smoke_config_has_bounded_runtime(self):
        config_path = Path(__file__).parents[1] / "configs" / "smoke.yaml"
        config = load_config(config_path)
        self.assertEqual(config["project"]["time_budget_minutes"], 30)
        self.assertFalse(config["embeddings"]["enabled"])
        self.assertEqual(len(config["_config_hash"]), 12)

    def test_invalid_runtime_and_search_values_are_rejected(self):
        cases = [
            ("negative top-k", "candidate_generation", "name_top_k", -1),
            ("reversed n-grams", "candidate_generation", "char_ngram_min", 7),
            ("invalid similarity", "candidate_generation", "min_name_similarity", 1.1),
            ("negative holdout", "training", "holdout_fraction", -0.1),
            ("zero trials", "training", "tuning_trials", 0),
            ("zero threshold steps", "training", "threshold_steps", 0),
            ("unknown device", "training", "device", "tpu"),
        ]
        for label, section, key, value in cases:
            with self.subTest(label=label):
                config = deepcopy(DEFAULT_CONFIG)
                config[section][key] = value
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_unknown_configuration_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text("training:\n  typo_trials: 3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training.typo_trials"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
