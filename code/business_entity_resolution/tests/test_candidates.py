import unittest

import pandas as pd

from ber.candidates import candidate_recall, generate_candidates_for_source
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


if __name__ == "__main__":
    unittest.main()
