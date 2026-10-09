"""Promote the candidate-history notebook's daily status/coverage outputs.

Compare complete division populations and pair snapshots, including entities
that disappeared. A per-day manifest commits the tables only after all writes
succeed. This is candidate history; candidate-feature/tile actions remain a
separate subsequent stage. Run one writer at a time.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from mtx.config import M2T_CANDIDATE_HISTORY_DIRECTORY
from mtx.candidates.transformer_candidates import PAIR_COLUMNS, validate_candidate_frames
from mtx.utils.daily_cache import (
    atomic_json, atomic_parquet, day_label, division_name, file_record,
    frame_signature, manifest_valid, read_json, signature, file_digest,
)

CANDIDATE_HISTORY_DIR = M2T_CANDIDATE_HISTORY_DIRECTORY
PAIR_KEY = ["meter_number", "candidate_transformer"]
GROUP_CONTEXT = ["tx_group_id", "tx_group_size", "tx_group_source", "meter_type"]


def _different(left, right):
    return left.astype("string").fillna("\x00").ne(right.astype("string").fillna("\x00"))


def compare_candidate_pairs(current, previous):
    """Outer pair-key comparison: NEW, MODIFIED, REMOVED or UNCHANGED."""
    for frame in (current, previous):
        if not set(PAIR_KEY).issubset(frame):
            raise ValueError("Candidate history requires the complete pair key.")
        if frame[PAIR_KEY].isna().any(axis=None) or frame.duplicated(PAIR_KEY).any():
            raise ValueError("Candidate history rejects null/duplicate pair keys.")
    columns = sorted((set(current) | set(previous)) - set(PAIR_KEY))
    left, right = current.copy(), previous.copy()
    # Preserve integral/text values during an outer join with added/removed rows.
    for column in columns:
        for frame in (left, right):
            if column not in frame:
                frame[column] = pd.NA
            frame[column] = frame[column].astype("string")
    merged = left.merge(right.rename(columns={c: f"previous_{c}" for c in columns}),
                        on=PAIR_KEY, how="outer", validate="one_to_one", indicator=True)
    changed = pd.DataFrame({c: _different(merged[c], merged[f"previous_{c}"]) for c in columns})
    both = merged["_merge"].eq("both")
    merged["current_exists"] = ~merged["_merge"].eq("right_only")
    merged["previous_exists"] = ~merged["_merge"].eq("left_only")
    merged["change_type"] = np.select(
        [merged["_merge"].eq("left_only"), merged["_merge"].eq("right_only"),
         both & changed.any(axis=1)], ["NEW", "REMOVED", "MODIFIED"], default="UNCHANGED",
    )
    def flag(name, fields):
        selected = [c for c in fields if c in changed]
        merged[name] = both & (changed[selected].any(axis=1) if selected else False)
    flag("coordinates_changed", ["meter_latitude", "meter_longitude", "transformer_latitude", "transformer_longitude"])
    flag("grouping_changed", GROUP_CONTEXT + ["group_signature"])
    flag("distance_changed", ["distance_ft"])
    flag("meter_context_changed", ["current_meter_floc", "meter_candidate_signature"])
    flag("transformer_context_changed", ["transformer_signature"])
    flag("division_changed", ["division"])
    flag("configuration_changed", ["configuration_signature"])
    if "division" in merged:
        merged["division"] = merged["division"].fillna(merged["previous_division"])
    for column in ("distance_ft", "previous_distance_ft"):
        if column in merged:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged.drop(columns="_merge").sort_values(PAIR_KEY).reset_index(drop=True)


def _list(value):
    return list(value) if isinstance(value, (list, tuple, np.ndarray)) else []


def _aggregate_signature(frame, key, sort_keys):
    if frame.empty:
        return {}
    result = {}
    for value, subset in frame.groupby(key, dropna=False, sort=False):
        result[value] = frame_signature(subset, sort_keys)
    return result


def _entity_status(current, previous, key, candidates_column):
    """Include removed meters/transformers, and compare their candidate sets."""
    columns = sorted((set(current) | set(previous)) - {key})
    left, right = current.copy(), previous.copy()
    for column in columns:
        for frame in (left, right):
            if column not in frame:
                frame[column] = pd.NA
    for frame in (left, right):
        for column in columns:
            if column != candidates_column:
                frame[column] = frame[column].astype("string")
    result = left.merge(right.rename(columns={c: f"previous_{c}" for c in columns}),
                        on=key, how="outer", validate="one_to_one", indicator=True)
    result["current_exists"] = ~result["_merge"].eq("right_only")
    result["previous_exists"] = ~result["_merge"].eq("left_only")
    current_sets = result[candidates_column].map(_list)
    previous_sets = result[f"previous_{candidates_column}"].map(_list)
    result[candidates_column] = current_sets
    result[f"previous_{candidates_column}"] = previous_sets
    result["new_candidates"] = [sorted(set(c) - set(p)) for c, p in zip(current_sets, previous_sets)]
    result["removed_candidates"] = [sorted(set(p) - set(c)) for c, p in zip(current_sets, previous_sets)]
    result["candidate_set_changed"] = [set(c) != set(p) for c, p in zip(current_sets, previous_sets)]
    changed = pd.Series(False, index=result.index)
    for column in columns:
        if column != candidates_column:
            changed |= _different(result[column], result[f"previous_{column}"])
    result["change_type"] = np.select(
        [result["_merge"].eq("left_only"), result["_merge"].eq("right_only"),
         result["_merge"].eq("both") & (changed | result["candidate_set_changed"])],
        ["NEW", "REMOVED", "MODIFIED"], default="UNCHANGED",
    )
    result["division"] = result["division"].fillna(result["previous_division"])
    return result.drop(columns="_merge").reset_index(drop=True)


def _meter_context(ami_df, groups):
    required = ["meter_number", "Division", "latitude", "longitude"]
    columns = required + (["current_meter_floc"] if "current_meter_floc" in ami_df else [])
    meters = ami_df[columns].rename(columns={"Division": "division", "latitude": "meter_latitude", "longitude": "meter_longitude"}).copy()
    if meters["meter_number"].duplicated().any() or meters["meter_number"].isna().any():
        raise ValueError("Meter history keys must be unique/non-null.")
    if "current_meter_floc" not in meters:
        meters["current_meter_floc"] = pd.Series(pd.NA, index=meters.index, dtype="string")
    groups = groups.rename(columns={"Division": "division"})
    if groups["meter_number"].duplicated().any() or set(groups["meter_number"]) != set(meters["meter_number"]):
        raise ValueError("Final grouping must exactly preserve the eligible AMI population.")
    meters = meters.merge(groups[["meter_number", "division", *GROUP_CONTEXT]],
                          on=["meter_number", "division"], how="left", validate="one_to_one")
    if meters["tx_group_source"].isna().any():
        raise ValueError("Grouping/AMI division mismatch in candidate history.")
    return meters


def _transformer_context(transformers):
    columns = ["transformer_floc", "division", "transformer_latitude", "transformer_longitude", "transformer_signature", "bank_unit_count"]
    if transformers.empty:
        return pd.DataFrame({c: pd.Series(dtype="string") for c in columns})
    rows = []
    for floc, units in transformers.groupby("transformer_floc", sort=True):
        if units["Division"].nunique() != 1 or units[["latitude", "longitude"]].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Transformer bank {floc} has conflicting division/coordinates.")
        row = units.iloc[0]
        keys = [c for c in ("sap_id", "gis_id", "transformer_floc") if c in units]
        rows.append({"transformer_floc": floc, "division": row["Division"],
                     "transformer_latitude": row["latitude"], "transformer_longitude": row["longitude"],
                     "transformer_signature": frame_signature(units, keys),
                     "bank_unit_count": int(units["sap_id"].nunique()) if "sap_id" in units else len(units)})
    return pd.DataFrame(rows, columns=columns)


def _contexts(pairs, meters, transformers, radius_ft):
    meter_sets = pairs.groupby("meter_number")["candidate_transformer"].agg(lambda values: sorted(set(values)))
    meter_hashes = _aggregate_signature(pairs, "meter_number", PAIR_KEY)
    meters = meters.copy()
    meters["candidate_transformers"] = meters["meter_number"].map(meter_sets).map(_list)
    meters["candidate_transformer_count"] = meters["candidate_transformers"].map(len)
    meters["meter_candidate_signature"] = meters["meter_number"].map(meter_hashes).fillna(signature("NO_CANDIDATES"))
    scope = meters["tx_group_id"].astype("string").fillna("standalone::" + meters["meter_number"].astype("string"))
    grouped = meters.assign(_scope=scope)
    hashes = _aggregate_signature(grouped, "_scope", ["meter_number"])
    meters["group_signature"] = scope.map(hashes)
    tx_sets = pairs.groupby("candidate_transformer")["meter_number"].agg(lambda values: sorted(set(values)))
    transformers = transformers.copy()
    transformers["candidate_meters"] = transformers["transformer_floc"].map(tx_sets).map(_list)
    transformers["candidate_meter_count"] = transformers["candidate_meters"].map(len)
    context = pairs.merge(meters.drop(columns=["division", *GROUP_CONTEXT, "candidate_transformers", "candidate_transformer_count"]),
                          on="meter_number", how="left", validate="many_to_one")
    context = context.merge(transformers.drop(columns=["division", "candidate_meters", "candidate_meter_count"]),
                            left_on="candidate_transformer", right_on="transformer_floc",
                            how="left", validate="many_to_one").drop(columns="transformer_floc")
    if context["transformer_signature"].isna().any():
        raise ValueError("Candidate transformer is absent from its division population.")
    context["configuration_signature"] = signature(1, radius_ft)
    return context, meters, transformers


def _finish_commit(root, day, manifest):
    summary = pd.read_parquet(root / "summary" / f"{day}.parquet")
    combined = root / "candidate_daily_summary.parquet"
    if combined.exists():
        existing = pd.read_parquet(combined)
        existing = existing.loc[existing["date"].ne(day)]
        summary_all = pd.concat([existing, summary], ignore_index=True)
    else:
        summary_all = summary
    atomic_parquet(summary_all.sort_values(["date", "division"]), combined)
    atomic_json({"date": day, "signature": manifest["signature"]}, root / "latest.json")
    return summary


def track_candidate_history(candidate_files, ami_df, grouping_df, transformers_df, *,
                            snapshot_date=None, radius_ft=550.0,
                            history_dir=CANDIDATE_HISTORY_DIR):
    """Return the daily division summary; pair deltas are in pair_changes/date/.

    Source and final-group context is compared through committed population
    snapshots. Raw source histories are retained by the driver independently;
    rejected coordinates therefore do not masquerade as candidate-stage inputs.
    Same-day changes after commit are rejected rather than silently reused.
    """
    day, root = day_label(snapshot_date), Path(history_dir)
    meters = _meter_context(ami_df, grouping_df)
    tx = _transformer_context(transformers_df)
    expected_divisions = set(meters["division"].unique())
    if set(candidate_files) != expected_divisions:
        raise ValueError("Candidate-history files must cover every eligible meter division.")
    fingerprint = signature(1, radius_ft, frame_signature(meters, ["meter_number"]),
                            frame_signature(tx, ["transformer_floc"]),
                            {d: file_digest(Path(p)) for d, p in sorted(candidate_files.items())})
    latest = read_json(root / "latest.json")
    committed = read_json(root / "manifests" / f"{day}.json")
    if latest and latest["date"] > day:
        raise ValueError("Candidate-history dates must advance.")
    if committed:
        if not manifest_valid(committed, root, fingerprint):
            raise ValueError("Today's committed candidate history is incomplete/corrupt or its inputs changed. Restore it or advance the date; do not overwrite history.")
        return _finish_commit(root, day, committed)
    if latest and latest["date"] == day:
        raise ValueError("Candidate-history commit pointer exists without its manifest.")
    # The manifest is the commit record; the convenience pointer/combined
    # summary may be missing after a crash. Recover the newest complete date.
    manifests = sorted((root / "manifests").glob("*.json"))
    previous_day = latest["date"] if latest else None
    if manifests and (not previous_day or manifests[-1].stem > previous_day):
        recovered_day = manifests[-1].stem
        recovered = read_json(manifests[-1])
        if not recovered or not manifest_valid(recovered, root, recovered["signature"]):
            raise ValueError("Previous candidate-history manifest is incomplete/corrupt.")
        if recovered_day >= day:
            raise ValueError("Candidate-history dates must advance.")
        _finish_commit(root, recovered_day, recovered)
        previous_day = recovered_day
    elif not manifests and (root / "candidate_daily_summary.parquet").exists():
        raise ValueError("Existing notebook history needs an explicit migration; use a separate daily history directory.")
    if previous_day:
        previous_manifest = read_json(root / "manifests" / f"{previous_day}.json")
        if not previous_manifest or not manifest_valid(previous_manifest, root, previous_manifest["signature"]):
            raise ValueError("Previous candidate-history commit is incomplete/corrupt; restore it before comparing.")
    old_meters = pd.read_parquet(root / "snapshots" / previous_day / "meters.parquet") if previous_day else meters.iloc[:0].copy()
    old_tx = pd.read_parquet(root / "snapshots" / previous_day / "transformers.parquet") if previous_day else tx.iloc[:0].copy()
    divisions = sorted(set(meters["division"]) | set(tx["division"]) |
                       set(old_meters["division"]) | set(old_tx["division"]))
    if not divisions:
        raise ValueError("No current or previous candidate-history population exists.")
    files, summaries, meter_snapshots, tx_snapshots = [], [], [], []
    for division in divisions:
        name = division_name(division)
        current_meters = meters.loc[meters["division"].eq(division)].copy()
        current_tx = tx.loc[tx["division"].eq(division)].copy()
        if division in candidate_files:
            path = Path(candidate_files[division])
            pairs = pd.read_parquet(path)
            validate_candidate_frames(pairs, grouping_df.rename(columns={"Division": "division"}).loc[
                grouping_df.rename(columns={"Division": "division"})["division"].eq(division)],
                pd.read_parquet(path.parent / "zero.parquet"), division=division,
                radius_ft=radius_ft, transformer_df=transformers_df)
        else:
            pairs = pd.DataFrame({c: pd.Series(dtype="string") for c in PAIR_COLUMNS})
            pairs["distance_ft"] = pd.Series(dtype="float64")
        context, current_meters, current_tx = _contexts(pairs, current_meters, current_tx, radius_ft)
        previous_pairs_path = root / "snapshots" / previous_day / f"{name}.parquet" if previous_day else None
        previous_pairs = pd.read_parquet(previous_pairs_path) if previous_pairs_path and previous_pairs_path.exists() else context.iloc[:0].copy()
        changes = compare_candidate_pairs(context, previous_pairs)
        meter_status = _entity_status(current_meters, old_meters.loc[old_meters["division"].eq(division)],
                                      "meter_number", "candidate_transformers")
        transformer_status = _entity_status(current_tx, old_tx.loc[old_tx["division"].eq(division)],
                                            "transformer_floc", "candidate_meters")
        zero = meter_status.loc[meter_status["current_exists"] & meter_status["candidate_transformers"].map(len).eq(0)].copy()
        zero["zero_candidate_reason"] = "NO_TRANSFORMERS_IN_DIVISION" if current_tx.empty else "NO_TRANSFORMER_WITHIN_RADIUS"
        unused = transformer_status.loc[transformer_status["current_exists"] & transformer_status["candidate_meters"].map(len).eq(0)]
        impacted = changes.loc[changes["change_type"].ne("UNCHANGED")]
        for folder, frame in {"meter_status": meter_status, "transformer_status": transformer_status,
                              "zero_candidate_meters": zero, "unused_transformers": unused,
                              "source_impacted_pairs": impacted, "pair_changes": changes}.items():
            output = root / folder / day / f"{name}.parquet"
            atomic_parquet(frame.assign(change_date=day), output)
            files.append(file_record(output, root))
        snapshot = root / "snapshots" / day / f"{name}.parquet"
        atomic_parquet(context, snapshot)
        files.append(file_record(snapshot, root))
        meter_snapshots.append(current_meters)
        tx_snapshots.append(current_tx)
        counts = changes["change_type"].value_counts()
        summaries.append({"date": day, "division": division, "radius_ft": radius_ft,
                          "ami_meter_count": len(current_meters), "candidate_pairs": len(context),
                          "meters_with_candidates": int(current_meters["candidate_transformer_count"].gt(0).sum()),
                          "meters_without_candidates": len(zero), "transformer_record_count": len(current_tx),
                          "unused_transformers": len(unused), "new_pairs": int(counts.get("NEW", 0)),
                          "removed_pairs": int(counts.get("REMOVED", 0)), "modified_pairs": int(counts.get("MODIFIED", 0)),
                          "unchanged_pairs": int(counts.get("UNCHANGED", 0)),
                          "new_meters": int(meter_status["change_type"].eq("NEW").sum()),
                          "removed_meters": int(meter_status["change_type"].eq("REMOVED").sum()),
                          "new_transformers": int(transformer_status["change_type"].eq("NEW").sum()),
                          "removed_transformers": int(transformer_status["change_type"].eq("REMOVED").sum())})
    for filename, frames in (("meters", meter_snapshots), ("transformers", tx_snapshots)):
        path = root / "snapshots" / day / f"{filename}.parquet"
        atomic_parquet(pd.concat(frames, ignore_index=True), path)
        files.append(file_record(path, root))
    summary = pd.DataFrame(summaries)
    path = root / "summary" / f"{day}.parquet"
    atomic_parquet(summary, path)
    files.append(file_record(path, root))
    manifest = {"complete": True, "version": 1, "signature": fingerprint,
                "date": day, "previous_date": previous_day, "files": files}
    atomic_json(manifest, root / "manifests" / f"{day}.json")
    return _finish_commit(root, day, manifest)


def get_division_candidate_history(division, *, history_dir=CANDIDATE_HISTORY_DIR):
    frame = pd.read_parquet(Path(history_dir) / "candidate_daily_summary.parquet")
    return frame.loc[frame["division"].eq(division)].sort_values("date").reset_index(drop=True)


def _committed_frames(folder, history_dir):
    root = Path(history_dir)
    frames = []
    for marker in sorted((root / "manifests").glob("*.json")):
        manifest = read_json(marker)
        if not manifest or manifest.get("complete") is not True:
            continue
        for entry in manifest["files"]:
            relative = Path(entry["path"])
            if relative.parts[0] != folder:
                continue
            path = root / relative
            if file_digest(path) != entry["sha256"]:
                raise ValueError(f"Committed candidate history is corrupt: {path}")
            frames.append(pd.read_parquet(path))
    return frames


def get_meter_candidate_history(meter_number, *, history_dir=CANDIDATE_HISTORY_DIR):
    frames = _committed_frames("meter_status", history_dir)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    return frame.loc[frame["meter_number"].astype(str).eq(str(meter_number))].reset_index(drop=True)


def get_transformer_candidate_history(transformer_floc, *, history_dir=CANDIDATE_HISTORY_DIR):
    frames = _committed_frames("transformer_status", history_dir)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    return frame.loc[frame["transformer_floc"].astype(str).eq(str(transformer_floc))].reset_index(drop=True)
