import unittest

import pandas as pd

from ber.candidates import generate_candidates_for_source
from ber.features import FEATURE_COLUMNS, build_pair_features, label_pair_features
from ber.metrics import entity_fbeta, macro_fbeta, optimize_threshold
from ber.normalization import prepare_source_frame


class MetricsAndFeatureTests(unittest.TestCase):
    def test_macro_metric_includes_singletons(self):
        truth = {"S1-1": {"S2-1"}, "S1-2": set()}
        self.assertEqual(macro_fbeta(truth, {"S1-1": {"S2-1"}}), 1.0)
        self.assertEqual(macro_fbeta(truth, {"S1-1": {"S2-1"}, "S1-2": {"S2-9"}}), 0.5)
        self.assertAlmostEqual(entity_fbeta({"a", "b"}, {"a", "x"}), 0.5)

    def test_features_and_threshold(self):
        columns = ["entity_id", "business_name", "business_address", "country"]
        queries = prepare_source_frame(pd.DataFrame([["S1-1", "Acme Ltd", "14 Main Rd", "US"]], columns=columns))
        corpus = prepare_source_frame(
            pd.DataFrame(
                [["S2-1", "Acme Limited", "14 Main Road", "US"], ["S2-2", "Other", "99 Side St", "US"]],
                columns=columns,
            )
        )
        config = {
            "name_top_k": 2, "address_top_k": 2, "max_candidates_per_query": 3,
            "max_exact_block_size": 10, "char_ngram_min": 3, "char_ngram_max": 5,
            "min_name_similarity": 0.0, "min_address_similarity": 0.0, "max_tfidf_features": 1000,
        }
        pairs = generate_candidates_for_source(queries, corpus, "S2", config)
        features = build_pair_features(pairs, queries, corpus)
        self.assertTrue(set(FEATURE_COLUMNS) <= set(features.columns))
        labeled = label_pair_features(features, {"S1-1": {"S2-1"}})
        self.assertEqual(int(labeled.loc[labeled.candidate_entity_id == "S2-1", "label"].iloc[0]), 1)
        scored = features[["source1_entity_id", "candidate_entity_id"]].copy()
        scored["score"] = [0.9 if value == "S2-1" else 0.1 for value in scored.candidate_entity_id]
        threshold, score, curve = optimize_threshold(scored, {"S1-1": {"S2-1"}}, 0.2, 0.8, 4)
        self.assertEqual(score, 1.0)
        self.assertFalse(curve.empty)


if __name__ == "__main__":
    unittest.main()
