from __future__ import annotations

import numpy as np
import pandas as pd


def assign_group_splits(
    source1_ids: pd.Series,
    validation_fraction: float,
    holdout_fraction: float,
    seed: int,
) -> pd.Series:
    key = f"{seed:016d}"[-16:]
    hashes = pd.util.hash_pandas_object(source1_ids.astype(str), index=False, hash_key=key).to_numpy(
        dtype=np.uint64
    )
    fractions = hashes.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)
    labels = np.full(len(source1_ids), "train", dtype=object)
    labels[fractions < holdout_fraction] = "holdout"
    labels[(fractions >= holdout_fraction) & (fractions < holdout_fraction + validation_fraction)] = "validation"
    return pd.Series(labels, index=source1_ids.index, name="split")
