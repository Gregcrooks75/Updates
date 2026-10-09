"""Daily meter-to-transformer grouping, candidates, and candidate history."""
from __future__ import annotations
from pathlib import Path
import geopandas as gpd
import pandas as pd

from mtx.config import PROJECT_ROOT, M2T_DIRECTORY, M2T_GROUP_HISTORY_DIRECTORY, M2T_CANDIDATE_HISTORY_DIRECTORY
from mtx.run_config import RunConfig
from mtx.ingestion.ami_cache import load_or_refresh_ny_ami
from mtx.ingestion.transformer_cache import load_or_refresh_ny_transformers
from mtx.ingestion.itron_group_cache import load_or_refresh_itron_groups
from mtx.ingestion.itron_groups import build_itron_tx_subgroups
from mtx.ingestion.adam_cache import load_or_refresh_adam_data
from mtx.history.ami_history import track_ami_history, HISTORY_DIR as AMI_HISTORY_DIR
from mtx.history.transformer_history import track_transformer_history, TRANSFORMER_HISTORY_DIR
from mtx.history.itron_history import track_grouping_history, ITRON_HISTORY_DIR
from mtx.history.adam_history import track_adam_history, get_adam_training_changes, ADAM_HISTORY_DIR
from mtx.history.candidate_history import track_candidate_history
from mtx.utils.daily_cache import atomic_json, day_label, frame_signature, read_json, signature
from meter_to_transformer.data_cleaning.standardize_inputs import standardize_all_inputs
from meter_to_transformer.data_cleaning.validate_coordinates import validate_coordinates, save_rejection_log
from meter_to_transformer.stages.run_meter_grouping import run_meter_grouping_stage
from meter_to_transformer.stages.run_transformer_candidates import run_transformer_candidate_stage
from meter_to_transformer.validation.validate_transformer_candidates import validate_transformer_candidates

CONFIG = RunConfig(
    week_start="2026-02-02", week_end="2026-02-08",
    correlation_start="2026-02-02", correlation_end="2026-02-08",
    meter_correlation_radius=550, transformer_candidate_radius_ft=550,
    correlation_method="spearman", group_source="itron",
    weekly_partition_size=25_000, seed_batch_size=5_000,
    transformer_candidate_batch_size=25_000,
    force_ami_refresh=False, force_transformer_refresh=False, force_itron_refresh=False,
    transformer_division_col="Division", overwrite_m2t_outputs=False,
)
NYSEG_DIVISION_FILE = PROJECT_ROOT / "data" / "data augmentation" / "NYSEG_Electric_Divisions.shp"
RGE_DIVISION_FILE = PROJECT_ROOT / "data" / "data augmentation" / "RGE_Electric_Divisions.shp"
ITRON_REQUIRED_COLUMNS = ("meter_number", "Tx_Subgroup_ID", "tx_group_id", "tx_group_size", "group_source", "group_type")


def _check_committed_source_day(frame, root, prefix, summary_name, previous_name, day, keys):
    """Existing trackers reuse a committed date; refuse newer same-day input."""
    root = Path(root)
    summary_path = root / summary_name
    if not summary_path.exists():
        return
    summary = pd.read_parquet(summary_path)
    if not pd.to_datetime(summary["date"]).dt.strftime("%Y-%m-%d").eq(day).any():
        return
    pending = root / f"pending_{prefix}_{day}.parquet"
    snapshot = pending if pending.exists() else root / previous_name
    saved = pd.read_parquet(snapshot)
    if set(saved.columns) != set(frame.columns) or frame_signature(saved, keys) != frame_signature(frame, keys):
        raise ValueError(f"{prefix}: {day} was already committed with different input. Daily histories are immutable; advance the date or explicitly migrate that history.")


def _pin_inputs(day, config, ami, transformers, itron, adam, divisions):
    polygon_frame = pd.DataFrame({"Division": divisions["Division"], "geometry_wkb": divisions.geometry.to_wkb(hex=True)})
    transformer_keys = [c for c in ("sap_id", "gis_id", "transformer_floc") if c in transformers]
    checks = [
        (ami, AMI_HISTORY_DIR, "ami", "ami_daily_summary.parquet", "previous_ami_snapshot.parquet", ["meter_number"]),
        (transformers, TRANSFORMER_HISTORY_DIR, "transformer", "transformer_daily_summary.parquet", "previous_transformer_snapshot.parquet", transformer_keys),
        (itron, ITRON_HISTORY_DIR, "grouping", "grouping_daily_summary.parquet", "previous_grouping_snapshot.parquet", ["meter_number"]),
        (adam, ADAM_HISTORY_DIR, "adam", "adam_daily_summary.parquet", "previous_adam_snapshot.parquet", ["meter_number"]),
    ]
    for frame, root, prefix, summary, previous, keys in checks:
        _check_committed_source_day(frame, root, prefix, summary, previous, day, keys)
    fingerprint = signature(1, config.group_source, config.transformer_candidate_radius_ft,
                            config.transformer_candidate_batch_size,
                            *(frame_signature(frame, keys) for frame, _, _, _, _, keys in checks),
                            frame_signature(polygon_frame, ["Division", "geometry_wkb"]))
    marker = M2T_DIRECTORY / "runs" / f"{day}.json"
    pinned = read_json(marker)
    if pinned and pinned.get("signature") != fingerprint:
        raise ValueError("Today's pipeline inputs/configuration changed after the run started. Resume with the pinned inputs or advance the date; do not overwrite committed daily history.")
    if not pinned:
        atomic_json({"date": day, "signature": fingerprint}, marker)


def main(config=CONFIG, *, snapshot_date=None, smartlink_cluster_root=None):
    day = day_label(snapshot_date)
    print(f"SMARTLINK METER-TO-TRANSFORMER PIPELINE: {day}")
    ami = load_or_refresh_ny_ami(force_refresh=config.force_ami_refresh)
    transformers = load_or_refresh_ny_transformers(force_refresh=config.force_transformer_refresh)
    itron = load_or_refresh_itron_groups(refresh_fn=build_itron_tx_subgroups,
                                        force_refresh=config.force_itron_refresh, max_age_days=1, cache_day=day,
                                        required_columns=ITRON_REQUIRED_COLUMNS)
    adam = load_or_refresh_adam_data(force_refresh=False)
    nyseg = gpd.read_file(NYSEG_DIVISION_FILE).rename(columns={"DIVISION": "Division"})
    rge = gpd.read_file(RGE_DIVISION_FILE).rename(columns={"DISTRICT": "Division"})
    if nyseg.crs is None or rge.crs is None:
        raise ValueError("Both division shapefiles must declare their source CRS.")
    nyseg, rge = nyseg.to_crs(epsg=4326), rge.to_crs(epsg=4326)
    divisions = gpd.GeoDataFrame(pd.concat([nyseg[["Division", "geometry"]], rge[["Division", "geometry"]]],
                                           ignore_index=True), geometry="geometry", crs="EPSG:4326")
    ami, transformers, itron, divisions, adam = standardize_all_inputs(ami, transformers, itron, divisions, adam)
    _pin_inputs(day, config, ami, transformers, itron, adam, divisions)
    # Preserve the existing source histories before coordinate rejection.
    ami_changes = track_ami_history(ami, snapshot_date=day)
    transformer_changes = track_transformer_history(transformers, snapshot_date=day)
    itron_changes = track_grouping_history(itron, snapshot_date=day)
    adam_changes = track_adam_history(adam, snapshot_date=day)
    adam_training_changes = get_adam_training_changes(snapshot_date=day)
    ami, invalid_meters = validate_coordinates(ami, divisions)
    transformers, invalid_transformers = validate_coordinates(transformers, divisions)
    save_rejection_log(invalid_meters, invalid_transformers, snapshot_date=day)
    # A rejected bank FLOC rejects the whole bank, as in the existing driver.
    bad_meters = set(invalid_meters["meter_number"].dropna())
    bad_transformers = set(invalid_transformers["transformer_floc"].dropna())
    ami = ami.loc[~ami["meter_number"].isin(bad_meters)].reset_index(drop=True)
    transformers = transformers.loc[~transformers["transformer_floc"].isin(bad_transformers)].reset_index(drop=True)
    itron = itron.loc[~itron["meter_number"].isin(bad_meters)].reset_index(drop=True)
    adam = adam.loc[~adam["meter_number"].isin(bad_meters)].reset_index(drop=True)
    group_files = run_meter_grouping_stage(config, ami, itron_groups_df=itron,
                                           smartlink_cluster_root=smartlink_cluster_root, snapshot_date=day)
    groups = pd.concat([pd.read_parquet(path) for path in group_files.values()], ignore_index=True) if group_files else pd.DataFrame(
        columns=["meter_number", "division", "tx_group_id", "tx_group_size", "tx_group_source", "meter_type"])
    group_changes = None
    if config.group_source == "itron" and len(groups):
        standardized_groups = groups.rename(columns={"division": "Division"})
        _check_committed_source_day(standardized_groups, M2T_GROUP_HISTORY_DIRECTORY, "grouping",
                                   "grouping_daily_summary.parquet", "previous_grouping_snapshot.parquet",
                                   day, ["meter_number"])
        # A separate root prevents mixing raw ITRON and final AMI-population schemas.
        group_changes = track_grouping_history(standardized_groups, snapshot_date=day,
                                               history_dir=M2T_GROUP_HISTORY_DIRECTORY, group_source="itron")
    candidate_files = run_transformer_candidate_stage(config, group_files, ami, transformers, snapshot_date=day)
    validate_transformer_candidates(config, candidate_files, group_files, transformer_df=transformers)
    candidate_summary = track_candidate_history(candidate_files, ami, groups, transformers,
                                                 snapshot_date=day, radius_ft=config.transformer_candidate_radius_ft)
    return {"snapshot_date": day, "ami_df": ami, "transformer_df": transformers,
            "itron_groups_df": itron, "adam_df": adam, "division_gdf": divisions,
            "ami_changes": ami_changes, "transformer_changes": transformer_changes,
            "itron_changes": itron_changes, "adam_changes": adam_changes,
            "adam_training_changes": adam_training_changes, "group_changes": group_changes,
            "invalid_meters": invalid_meters, "invalid_transformers": invalid_transformers,
            "group_files": group_files, "candidate_files": candidate_files,
            "zero_candidate_files": {d: p.parent / "zero.parquet" for d, p in candidate_files.items()},
            "candidate_summary": candidate_summary,
            "pair_change_files": {d: M2T_CANDIDATE_HISTORY_DIRECTORY / "pair_changes" / day / f"{d}.parquet"
                                  for d in candidate_summary["division"]}}


if __name__ == "__main__":
    main()
