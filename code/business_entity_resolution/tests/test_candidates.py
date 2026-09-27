import unittest

import pandas as pd

from ber.candidates import candidate_recall, exact_block, generate_candidates_for_source
from ber.normalization import prepare_source_frame


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "name_top_k": 3,
            "address_top_k": 3,
            "max_candidates_per_query": 6,
            "max_exact_block_size": 20,
            "char_ngram_min": 3,
            "char_ngram_max": 5,
            "min_name_similarity": 0.1,
            "min_address_similarity": 0.1,
            "max_tfidf_features": 5000,
        }

    def test_multilane_retrieval_and_recall(self):
        queries = prepare_source_frame(
            pd.DataFrame(
                [
                    ["S1-1", "Acme Technologies Ltd", "14 Main Road 282005", "India"],
                    ["S1-2", "Lakshmi Healthcare", "7 Park Street 560001", "India"],
                ],
                columns=["entity_id", "business_name", "business_address", "country"],
            )
        )
        corpus = prepare_source_frame(
            pd.DataFrame(
                [
                    ["S2-1", "Acme Technology Limited", "14 Main Rd 282005", "India"],
                    ["S2-2", "लक्ष्मी हेल्थकेयर", "7 Park Street 560001", "India"],
                    ["S2-3", "Different Company", "99 Other Avenue", "India"],
                ],
                columns=["entity_id", "business_name", "business_address", "country"],
            )
        )
        pairs = generate_candidates_for_source(queries, corpus, "S2", self.config)
        pair_set = set(zip(pairs.source1_entity_id, pairs.candidate_entity_id))
        self.assertIn(("S1-1", "S2-1"), pair_set)
        self.assertIn(("S1-2", "S2-2"), pair_set)
        recall = candidate_recall(pairs, {"S1-1": {"S2-1"}, "S1-2": {"S2-2"}})
        self.assertEqual(recall["pair_recall"], 1.0)

    def test_exact_name_does_not_hide_fuzzy_match(self):
        self.config["tfidf_query_chunk_size"] = 1
        queries = prepare_source_frame(
            pd.DataFrame(
                [["S1-1", "Alpha Bakery", "10 Main Road", "India"]],
                columns=["entity_id", "business_name", "business_address", "country"],
            )
        )
        corpus = prepare_source_frame(
            pd.DataFrame(
                [
                    ["S2-1", "Alpha Bakery", "99 Other Street", "India"],
                    ["S2-2", "Alpha Bakeries", "10 Main Rd", "India"],
                ],
                columns=["entity_id", "business_name", "business_address", "country"],
            )
        )
        pairs = generate_candidates_for_source(queries, corpus, "S2", self.config)
        self.assertIn("S2-2", set(pairs["candidate_entity_id"]))

    def test_common_query_key_does_not_expand_exact_join(self):
        columns = ["entity_id", "name_norm"]
        queries = pd.DataFrame(
            [["S1-1", "common shop"], ["S1-2", "common shop"], ["S1-3", "common shop"]],
            columns=columns,
        )
        corpus = pd.DataFrame([["S2-1", "common shop"]], columns=columns)
        lane = exact_block(queries, corpus, "name_norm", "S2", "exact_name", max_block_size=2)
        self.assertTrue(lane.empty)

    def test_token_hash_finds_cross_script_address_evidence(self):
        self.config.update(
            {
                "method": "token_hash",
                "token_top_k": 3,
                "token_min_similarity": 0.01,
                "token_hash_features": 4096,
                "token_query_chunk_size": 1,
                "token_threads": 1,
                "max_tfidf_document_frequency": 100,
            }
        )
        columns = ["entity_id", "business_name", "business_address", "country"]
        queries = prepare_source_frame(pd.DataFrame(
            [["S1-1", "लक्ष्मी हेल्थकेयर", "7 Park Street 560001", "India"]], columns=columns
        ))
        corpus = prepare_source_frame(pd.DataFrame(
            [
                ["S2-1", "Lakshmi Healthcare", "7 Park Road 560001", "India"],
                ["S2-2", "Unrelated Shop", "99 Other Avenue", "India"],
            ], columns=columns
        ))
        pairs = generate_candidates_for_source(queries, corpus, "S2", self.config)
        match = pairs.loc[pairs["candidate_entity_id"].eq("S2-1")]
        self.assertEqual(len(match), 1)
        self.assertGreater(float(match["token_tfidf"].iloc[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
