"""Candidate integrity and explicit zero-candidate population accounting."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


from mtx.candidates.transformer_candidates import validate_candidate_frames


def validate_transformer_candidates(config, candidate_files, group_files, *, transformer_df=None):
    if set(candidate_files) != set(group_files):
        raise AssertionError("Candidate and grouping divisions differ.")
    for division, path in candidate_files.items():
        validate_candidate_frames(pd.read_parquet(path), pd.read_parquet(group_files[division]),
                                  pd.read_parquet(Path(path).parent / "zero.parquet"),
                                  division=division, radius_ft=config.transformer_candidate_radius_ft,
                                  transformer_df=transformer_df)
    print("TRANSFORMER CANDIDATE VALIDATION PASSED")
