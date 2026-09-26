import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ber.embeddings import dense_candidates_from_arrays
from ber.pipeline import _semantic_rerank_lane


class EmbeddingCandidateTests(unittest.TestCase):
    def test_dense_candidates_use_cosine_order(self):
        pairs = dense_candidates_from_arrays(
            np.asarray(["S1-1", "S1-2"]),
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.asarray(["S2-1", "S2-2", "S2-3"]),
            np.asarray([[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0]], dtype=np.float32),
            "S2",
            top_k=1,
        )
        mapping = dict(zip(pairs.source1_entity_id, pairs.candidate_entity_id))
        self.assertEqual(mapping, {"S1-1": "S2-1", "S1-2": "S2-2"})

    def test_semantic_rerank_is_bounded_to_cross_script_pairs(self):
        queries = pd.DataFrame(
            {"entity_id": ["S1-1", "S1-2"], "name_script": ["LATIN", "LATIN"]}
        )
        candidates = pd.DataFrame(
            {"entity_id": ["S2-1", "S2-2"], "name_script": ["DEVANAGARI", "LATIN"]}
        )
        pairs = pd.DataFrame(
            {
                "source1_entity_id": ["S1-1", "S1-2"],
                "candidate_entity_id": ["S2-1", "S2-2"],
                "target_source": ["S2", "S2"],
                "retrieval_score": [0.8, 0.9],
            }
        )
        config = {
            "embeddings": {
                "model_path": None,
                "model_name": "dummy",
                "batch_size_per_gpu": 2,
                "max_length": 8,
                "cross_script_only": True,
            },
            "candidate_generation": {"semantic_pairs_per_query": 2},
        }
        embedding_parts = [
            (np.asarray(["S1-1"]), np.asarray([[1.0, 0.0]], dtype=np.float32)),
            (np.asarray(["S2-1"]), np.asarray([[0.8, 0.2]], dtype=np.float32)),
        ]
        with tempfile.TemporaryDirectory() as directory, patch(
            "ber.pipeline._gpu_count", return_value=2
        ), patch("ber.pipeline.launch_parallel_encoding") as launch, patch(
            "ber.pipeline.load_embeddings", side_effect=embedding_parts
        ):
            stale_root = Path(directory) / "train" / "india" / "source2" / "output"
            stale_root.mkdir(parents=True)
            (stale_root / "_SUCCESS.rank-00").write_text("stale", encoding="utf-8")
            lane = _semantic_rerank_lane(
                queries,
                candidates,
                pairs,
                Path(directory),
                "train",
                "india",
                "source2",
                config,
                timeout_seconds=10,
            )
            self.assertEqual(launch.call_count, 2)
            self.assertFalse((stale_root / "_SUCCESS.rank-00").exists())
        self.assertEqual(lane["source1_entity_id"].tolist(), ["S1-1"])
        self.assertAlmostEqual(float(lane["dense_score"].iloc[0]), 0.8, places=5)


if __name__ == "__main__":
    unittest.main()
