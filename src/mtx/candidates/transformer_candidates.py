"""Canonical BallTree candidates, with the notebook's division/batch layout."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from mtx.grouping.group_assignments import GROUP_COLUMNS
from mtx.utils.daily_cache import (
    atomic_json, atomic_parquet, file_record, frame_signature,
    manifest_valid, merge_parquet_parts, read_json, signature, clean_temporary_files,
)

EARTH_RADIUS_FT = 20_902_231.0
PAIR_COLUMNS = GROUP_COLUMNS + ["candidate_transformer", "distance_ft"]
DIVISION_ALIASES = {
    "Central/Rochester": "Central", "Lakeshore/Sodus": "Sodus",
    "Genesee Valley/Filmore": "Fillmore",
}


def validcoordinates(df, lat_col, lon_col):
    out = df.copy()
    for column in (lat_col, lon_col):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    mask = out[lat_col].between(-90, 90) & out[lon_col].between(-180, 180)
    return out.loc[mask].reset_index(drop=True)


def _aliases(frame, aliases):
    return frame.rename(columns={old: new for old, new in aliases.items()
                                 if old in frame and new not in frame}).copy()


def _identifiers(frame, columns):
    for column in columns:
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or frame[column].eq("").any():
            raise ValueError(f"Missing or blank candidate input key: {column}")
    return frame


def prepare_candidate_inputs(meter_groups, ami_df, transformer_df, division,
                             transformer_division_col="Division"):
    """Accept standardized inputs; legacy aliases adapt to the same algorithm."""
    groups = _aliases(meter_groups, {"Division": "division"})
    ami = _aliases(ami_df, {"MeterID": "meter_number", "LATITUDE": "latitude",
                           "LONGITUDE": "longitude", "division": "Division"})
    tx = _aliases(transformer_df, {"FuncLoc": "transformer_floc",
                                  "Location_Y": "latitude", "Location_X": "longitude",
                                  transformer_division_col: "Division", "DivName": "Division"})
    for name, frame, required in (
        ("groups", groups, set(GROUP_COLUMNS)),
        ("AMI", ami, {"meter_number", "latitude", "longitude"}),
        ("transformers", tx, {"transformer_floc", "Division", "latitude", "longitude"}),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing candidate columns: {sorted(missing)}")
    groups = _identifiers(groups, ["meter_number", "division"])
    ami = _identifiers(ami, ["meter_number"])
    tx = _identifiers(tx, ["transformer_floc", "Division"])
    groups["division"] = groups["division"].replace(DIVISION_ALIASES)
    tx["Division"] = tx["Division"].replace(DIVISION_ALIASES)
    if groups["meter_number"].duplicated().any() or ami["meter_number"].duplicated().any():
        raise ValueError("Each meter must occur once before candidate generation.")
    groups = groups.loc[groups["division"].eq(division), GROUP_COLUMNS]
    if "Division" in ami:
        ami["Division"] = ami["Division"].astype("string").str.strip().replace(DIVISION_ALIASES)
        relevant = ami.loc[ami["meter_number"].isin(groups["meter_number"])]
        if not relevant["Division"].eq(division).all():
            raise ValueError(f"AMI/group division mismatch for {division}.")
    meters = groups.merge(ami[["meter_number", "latitude", "longitude"]],
                          on="meter_number", how="left", validate="one_to_one")
    valid_meters = validcoordinates(meters, "latitude", "longitude")
    if len(valid_meters) != len(meters):
        raise ValueError("Reject invalid or missing meter coordinates before the candidate stage.")
    tx = tx.loc[tx["Division"].eq(division),
                ["transformer_floc", "Division", "latitude", "longitude"]].copy()
    for column in ("latitude", "longitude"):
        tx[column] = pd.to_numeric(tx[column], errors="coerce")
    conflicting = tx.groupby("transformer_floc")[["latitude", "longitude"]].nunique(dropna=False)
    if conflicting.gt(1).any(axis=None):
        raise ValueError(f"Transformer bank has conflicting coordinates in {division}.")
    if len(validcoordinates(tx, "latitude", "longitude")) != len(tx):
        raise ValueError("Reject invalid transformer coordinates before candidate generation.")
    tx = tx.drop_duplicates("transformer_floc").sort_values("transformer_floc").reset_index(drop=True)
    meters = valid_meters.sort_values("meter_number").reset_index(drop=True)
    return meters, tx


def _empty_candidates():
    columns = {column: pd.Series(dtype="string") for column in PAIR_COLUMNS}
    columns["tx_group_size"] = pd.Series(dtype="int64")
    columns["distance_ft"] = pd.Series(dtype="float64")
    return pd.DataFrame(columns)


def _batch_candidates(meters, tx, tree, radius_ft):
    if meters.empty or tx.empty:
        counts = np.zeros(len(meters), dtype=int)
        candidates = _empty_candidates()
    else:
        indices, distances = tree.query_radius(
            np.radians(meters[["latitude", "longitude"]].to_numpy(dtype=float)),
            r=radius_ft / EARTH_RADIUS_FT, return_distance=True, sort_results=True,
        )
        counts = np.fromiter((len(value) for value in indices), dtype=int, count=len(meters))
        if counts.sum():
            candidates = meters.iloc[np.repeat(np.arange(len(meters)), counts)][GROUP_COLUMNS].reset_index(drop=True)
            candidates["candidate_transformer"] = tx["transformer_floc"].to_numpy()[np.concatenate(indices)]
            candidates["distance_ft"] = np.concatenate(distances) * EARTH_RADIUS_FT
            for column in PAIR_COLUMNS:
                if column not in {"tx_group_size", "distance_ft"}:
                    candidates[column] = candidates[column].astype("string")
            candidates["tx_group_size"] = candidates["tx_group_size"].astype("int64")
        else:
            candidates = _empty_candidates()
    zero = meters.loc[counts == 0, GROUP_COLUMNS].copy().reset_index(drop=True)
    for column in GROUP_COLUMNS:
        if column != "tx_group_size":
            zero[column] = zero[column].astype("string")
    zero["tx_group_size"] = zero["tx_group_size"].astype("int64")
    zero["zero_candidate_reason"] = pd.Series(
        ["NO_TRANSFORMERS_IN_DIVISION" if tx.empty else "NO_TRANSFORMER_WITHIN_RADIUS"] * len(zero),
        dtype="string",
    )
    return candidates, zero


def find_transformer_candidates(meter_groups, ami_df, transformer_df, division,
                                radius_ft=550.0, transformer_division_col="Division"):
    """Existing in-memory API, now using the standardized input contract."""
    if not np.isfinite(radius_ft) or radius_ft <= 0:
        raise ValueError("radius_ft must be finite and positive.")
    meters, tx = prepare_candidate_inputs(meter_groups, ami_df, transformer_df, division,
                                         transformer_division_col)
    tree = BallTree(np.radians(tx[["latitude", "longitude"]].to_numpy(dtype=float)),
                    metric="haversine") if len(tx) else None
    return _batch_candidates(meters, tx, tree, radius_ft)[0]


def build_transformer_candidate_partitions(meter_groups, ami_df, transformer_df, *,
                                          division, output_dir, radius_ft=550.0,
                                          batch_size=25_000, overwrite=False,
                                          transformer_division_col="Division"):
    """Persist every completed batch, including empty results, then commit the division.

    The final candidates.parquet preserves existing downstream APIs. Batch manifests
    permit restart; a division is reusable only after its final manifest is complete.
    Run one writer at a time. Batch filenames never contain meter/transformer IDs.
    """
    if not np.isfinite(radius_ft) or radius_ft <= 0 or batch_size <= 0:
        raise ValueError("A finite positive radius and positive batch_size are required.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    clean_temporary_files(root)
    meters, tx = prepare_candidate_inputs(meter_groups, ami_df, transformer_df, division,
                                         transformer_division_col)
    fingerprint = signature(1, division, float(radius_ft), batch_size,
                            frame_signature(meters, ["meter_number"]),
                            frame_signature(tx, ["transformer_floc"]))
    complete_file = root / "manifest.json"
    complete = read_json(complete_file)
    output = root / "candidates.parquet"
    if not overwrite and manifest_valid(complete, root, fingerprint):
        return output
    if complete and complete.get("signature") != fingerprint and not overwrite:
        raise ValueError(f"{division}: today's candidate inputs/configuration changed; use a new date or explicit overwrite before history commits.")
    complete_file.unlink(missing_ok=True)
    tree = BallTree(np.radians(tx[["latitude", "longitude"]].to_numpy(dtype=float)),
                    metric="haversine") if len(tx) else None
    parts, zero_parts, batch_records = [], [], []
    for number, start in enumerate(range(0, max(len(meters), 1), batch_size)):
        batch = meters.iloc[start:start + batch_size]
        batch_fingerprint = signature(fingerprint, number)
        part = root / f"p{number:05d}.parquet"
        zero_part = root / f"z{number:05d}.parquet"
        marker = root / f"b{number:05d}.json"
        saved = read_json(marker)
        if overwrite or not manifest_valid(saved, root, batch_fingerprint):
            candidates, zero = _batch_candidates(batch, tx, tree, radius_ft)
            atomic_parquet(candidates, part)
            atomic_parquet(zero, zero_part)
            saved = {"complete": True, "signature": batch_fingerprint,
                     "meter_count": len(batch), "candidate_pairs": len(candidates),
                     "zero_candidate_meters": len(zero),
                     "files": [file_record(part, root), file_record(zero_part, root)]}
            atomic_json(saved, marker)
        parts.append(part)
        zero_parts.append(zero_part)
        batch_records.append(saved)
    merge_parquet_parts(parts, output)
    zero_output = root / "zero.parquet"
    atomic_parquet(pd.concat([pd.read_parquet(path) for path in zero_parts], ignore_index=True), zero_output)
    # The stage wrapper validates pair integrity and population conservation before
    # this complete manifest is published, via the same dataframe validator.
    validate_candidate_frames(pd.read_parquet(output), meters[GROUP_COLUMNS],
                              pd.read_parquet(zero_output), division=division,
                              radius_ft=radius_ft, transformer_df=tx)
    files = [file_record(output, root), file_record(zero_output, root)]
    for number, saved in enumerate(batch_records):
        files.extend(saved["files"])
        files.append(file_record(root / f"b{number:05d}.json", root))
    atomic_json({"complete": True, "version": 1, "signature": fingerprint,
                 "division": division, "radius_ft": radius_ft, "batch_size": batch_size,
                 "meter_count": len(meters), "batch_count": len(parts),
                 "candidate_pairs": sum(item["candidate_pairs"] for item in batch_records),
                 "zero_candidate_meters": sum(item["zero_candidate_meters"] for item in batch_records),
                 "files": files}, complete_file)
    return output


def validate_candidate_frames(candidates, groups, zero, *, division, radius_ft,
                              transformer_df=None):
    required = {"meter_number", "division", "candidate_transformer", "distance_ft",
                "tx_group_id", "tx_group_size", "tx_group_source", "meter_type"}
    missing = required - set(candidates.columns)
    if missing:
        raise AssertionError(f"{division}: missing candidate columns: {sorted(missing)}")
    for name, frame, keys in (
        ("candidates", candidates, ["meter_number", "candidate_transformer"]),
        ("groups", groups, ["meter_number"]), ("zero", zero, ["meter_number"]),
    ):
        if not set(keys + ["division"]).issubset(frame.columns):
            raise AssertionError(f"{division}: {name} is missing keys/division.")
        if frame[keys].isna().any(axis=None) or frame[keys].astype("string").eq("").any(axis=None):
            raise AssertionError(f"{division}: null/blank {name} keys.")
        if frame.duplicated(keys).any():
            raise AssertionError(f"{division}: duplicate {name} keys.")
        if not frame["division"].eq(division).all():
            raise AssertionError(f"{division}: cross-division {name} records.")
    distances = pd.to_numeric(candidates["distance_ft"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(distances).all() or (distances < 0).any() or (distances > radius_ft + 0.1).any():
        raise AssertionError(f"{division}: invalid/out-of-radius candidate distance.")
    population = set(groups["meter_number"].astype(str))
    represented = set(candidates["meter_number"].astype(str))
    empty = set(zero["meter_number"].astype(str))
    if represented & empty or represented | empty != population:
        raise AssertionError(f"{division}: candidates plus zero ledger must exactly cover the input population.")
    if "zero_candidate_reason" not in zero or zero["zero_candidate_reason"].isna().any():
        raise AssertionError(f"{division}: zero-candidate reasons are required.")
    if len(candidates):
        expected = groups.set_index("meter_number")
        for column in ("division", "tx_group_id", "tx_group_size", "tx_group_source", "meter_type"):
            observed = candidates[column].astype("string").fillna("<NULL>").reset_index(drop=True)
            wanted = candidates["meter_number"].map(expected[column]).astype("string").fillna("<NULL>").reset_index(drop=True)
            if not observed.equals(wanted):
                raise AssertionError(f"{division}: candidate grouping context differs for {column}.")
    if transformer_df is not None:
        tx = transformer_df
        if "Division" in tx:
            tx = tx.loc[tx["Division"].eq(division)]
        allowed = set(tx["transformer_floc"].astype(str))
        if not set(candidates["candidate_transformer"].astype(str)).issubset(allowed):
            raise AssertionError(f"{division}: candidate transformer is outside its division population.")

