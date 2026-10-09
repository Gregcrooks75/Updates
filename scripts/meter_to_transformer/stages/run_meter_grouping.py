"""Daily grouping cache around the existing assignment implementations."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from mtx.config import M2T_GROUP_DIRECTORY
from mtx.grouping.group_assignments import (
    GROUP_COLUMNS, build_itron_group_assignments, build_smartlink_group_assignments,
)
from mtx.utils.daily_cache import (
    atomic_json, atomic_parquet, day_label, division_name, file_record,
    frame_signature, manifest_valid, read_json, signature, clean_temporary_files,
)

OUTPUT_DIR = M2T_GROUP_DIRECTORY


def _population(ami_df):
    ami = ami_df.rename(columns={"MeterID": "meter_number", "Division": "division"}).copy()
    if not {"meter_number", "division"}.issubset(ami):
        raise ValueError("AMI is missing meter_number/division.")
    ami = ami[["meter_number", "division"]]
    for column in ami:
        ami[column] = ami[column].astype("string").str.strip()
        if ami[column].isna().any() or ami[column].eq("").any():
            raise ValueError(f"Grouping population has missing {column}.")
    if ami["meter_number"].duplicated().any():
        raise ValueError("Resolve duplicate AMI meters before grouping.")
    return ami


def _validate_groups(groups, population, division):
    if list(groups.columns) != GROUP_COLUMNS:
        raise ValueError("Grouping output schema differs from GROUP_COLUMNS.")
    if groups["meter_number"].isna().any() or groups["meter_number"].duplicated().any():
        raise ValueError(f"{division}: grouping meter keys must be unique/non-null.")
    if set(groups["meter_number"]) != set(population["meter_number"]):
        raise ValueError(f"{division}: grouping must preserve the complete AMI population.")
    if not groups["division"].eq(division).all():
        raise ValueError(f"{division}: grouping crosses divisions.")
    has_group = groups["tx_group_id"].notna()
    expected = groups.loc[has_group].groupby("tx_group_id")["meter_number"].transform("size")
    if not groups.loc[has_group, "tx_group_size"].eq(expected).all():
        raise ValueError(f"{division}: group sizes disagree with actual population.")
    if not groups.loc[~has_group, "tx_group_size"].eq(1).all():
        raise ValueError(f"{division}: ungrouped meters must have size one.")
    wanted_type = groups["tx_group_size"].gt(1).map({True: "clustered", False: "standalone"})
    if not groups["meter_type"].eq(wanted_type).all():
        raise ValueError(f"{division}: grouping meter_type disagrees with group size.")


def run_meter_grouping_stage(config, ami_df, itron_groups_df=None,
                            smartlink_cluster_root=None, *, snapshot_date=None,
                            output_dir=OUTPUT_DIR):
    day = day_label(snapshot_date)
    population = _population(ami_df)
    outputs = {}
    for division in sorted(population["division"].unique()):
        subset = population.loc[population["division"].eq(division)].copy()
        root = Path(output_dir) / day / division_name(division)
        clean_temporary_files(root)
        output, marker = root / "groups.parquet", root / "manifest.json"
        if config.group_source == "itron":
            if itron_groups_df is None:
                raise ValueError("group_source='itron' requires itron_groups_df.")
            source = itron_groups_df.loc[itron_groups_df["meter_number"].isin(subset["meter_number"])].copy()
            source_signature = frame_signature(source, ["meter_number", "Tx_Subgroup_ID"])
        elif config.group_source == "smartlink":
            if smartlink_cluster_root is None:
                raise ValueError("group_source='smartlink' requires smartlink_cluster_root.")
            cluster_path = Path(smartlink_cluster_root) / f"{division}_meters_in_clusters.parquet"
            standalone_path = Path(smartlink_cluster_root) / f"{division}_standalone_meters.parquet"
            clustered, standalone = pd.read_parquet(cluster_path), pd.read_parquet(standalone_path)
            for frame in (clustered, standalone):
                frame["meter_number"] = frame["meter_number"].astype("string").str.strip()
            source_signature = signature(frame_signature(clustered, ["meter_number"]),
                                         frame_signature(standalone, ["meter_number"]))
        else:
            raise ValueError(f"Unsupported grouping source: {config.group_source}")
        fingerprint = signature(1, config.group_source, division,
                                frame_signature(subset, ["meter_number"]), source_signature)
        saved = read_json(marker)
        if not config.overwrite_m2t_outputs and manifest_valid(saved, root, fingerprint):
            _validate_groups(pd.read_parquet(output), subset, division)
            outputs[division] = output
            continue
        if saved and saved.get("signature") != fingerprint and not config.overwrite_m2t_outputs:
            raise ValueError(f"{division}: today's grouping inputs changed; use a new date or explicit overwrite before history commits.")
        marker.unlink(missing_ok=True)
        if config.group_source == "itron":
            groups = build_itron_group_assignments(subset, source)
        else:
            groups = build_smartlink_group_assignments(clustered, standalone, division)
        _validate_groups(groups, subset, division)
        atomic_parquet(groups, output)
        atomic_json({"complete": True, "version": 1, "signature": fingerprint,
                     "date": day, "division": division, "meter_count": len(groups),
                     "files": [file_record(output, root)]}, marker)
        outputs[division] = output
    return outputs
