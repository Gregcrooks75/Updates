"""Connect daily validated grouping files to the canonical candidate builder."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from mtx.config import M2T_CANDIDATE_DIRECTORY
from mtx.candidates.transformer_candidates import build_transformer_candidate_partitions
from mtx.utils.daily_cache import day_label, division_name

OUTPUT_DIR = M2T_CANDIDATE_DIRECTORY


def run_transformer_candidate_stage(config, group_files, ami_df, transformer_df, *,
                                    snapshot_date=None, output_dir=OUTPUT_DIR):
    day = day_label(snapshot_date)
    if ami_df["meter_number"].duplicated().any():
        raise ValueError("Candidate AMI meter keys must be unique across divisions.")
    if set(group_files) != set(ami_df["Division"].dropna().unique()):
        raise ValueError("Grouping files must cover every eligible AMI division.")
    outputs = {}
    for division, group_file in sorted(group_files.items()):
        groups = pd.read_parquet(group_file)
        population = ami_df.loc[ami_df["Division"].eq(division), "meter_number"]
        if set(groups["meter_number"]) != set(population):
            raise ValueError(f"{division}: grouping and candidate-stage AMI populations differ.")
        outputs[division] = build_transformer_candidate_partitions(
            groups, ami_df, transformer_df, division=division,
            output_dir=Path(output_dir) / day / division_name(division),
            radius_ft=config.transformer_candidate_radius_ft,
            batch_size=config.transformer_candidate_batch_size,
            overwrite=config.overwrite_m2t_outputs,
            transformer_division_col=config.transformer_division_col,
        )
    return outputs
