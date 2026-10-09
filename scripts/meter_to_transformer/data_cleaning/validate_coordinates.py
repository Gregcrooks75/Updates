from __future__ import annotations

import os
from uuid import uuid4
from mtx.utils.daily_cache import atomic_parquet, day_label

from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from datetime import date
from pathlib import Path

import pandas as pd





PROJECT_ROOT = Path(__file__).resolve().parents[3]

REJECTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "rejected"
)

REJECTED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
HISTORY_FILE = (
    REJECTED_DIR
    / "rejection_history.parquet"
)

import geopandas as gpd


def validate_coordinates(df, division_gdf):
    """Validate assigned-division membership without multiplying asset rows."""
    import numpy as np

    required = {"Division", "latitude", "longitude"}
    if not required.issubset(df.columns):
        raise ValueError("Coordinate validation requires Division/latitude/longitude.")
    if division_gdf.crs is None:
        raise ValueError("Division polygons must declare their source CRS.")
    divisions = division_gdf[["Division", "geometry"]].to_crs(epsg=4326)
    # Multiple polygon pieces of the same named division form one territory.
    divisions = divisions.dissolve(by="Division", as_index=False)
    if divisions["Division"].duplicated().any():
        raise ValueError("Division geometries must have one row per division.")
    out = pd.DataFrame(df.drop(columns=["geometry", "geometry_meter", "geometry_division"], errors="ignore")).copy()
    for column in ("latitude", "longitude"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    numeric_valid = (out["latitude"].between(-90, 90) & out["longitude"].between(-180, 180)
                     & np.isfinite(out["latitude"]) & np.isfinite(out["longitude"]))
    points = gpd.GeoDataFrame(out.copy(), geometry=gpd.points_from_xy(out["longitude"], out["latitude"]), crs="EPSG:4326")
    points = points.merge(divisions, on="Division", how="left", validate="many_to_one",
                          suffixes=("_meter", "_division"))
    inside = points["geometry_meter"].within(points["geometry_division"])
    key_valid = pd.Series(True, index=points.index)
    for key in ("meter_number", "transformer_floc"):
        if key in points:
            key_valid &= points[key].notna() & points[key].astype("string").str.strip().ne("")
    valid = numeric_valid.reset_index(drop=True) & inside.fillna(False).reset_index(drop=True) & key_valid
    points["coordinate_valid"] = valid
    points["rejection_reason"] = np.select(
        [~numeric_valid.reset_index(drop=True), ~key_valid, points["geometry_division"].isna(), ~inside.fillna(False)],
        ["INVALID_COORDINATES", "MISSING_ASSET_KEY", "MISSING_DIVISION_POLYGON", "OUTSIDE_ASSIGNED_DIVISION"],
        default="",
    )
    # Return ordinary dataframes; geometry diagnostics must not leak into tabular caches.
    result = pd.DataFrame(points.drop(columns=["geometry_meter", "geometry_division"]))
    return (result.loc[valid].reset_index(drop=True), result.loc[~valid].reset_index(drop=True))


def save_rejection_log(
    invalid_meters,
    invalid_transformers,
    *,
    snapshot_date=None,
):
    """
    Save:
        1. Historical rejection IDs (Parquet)
        2. Daily review file (Excel)

    Skip if today's files already exist.
    """

    today = day_label(snapshot_date)

    excel_file = (
        REJECTED_DIR
        / f"rejections_{today}.xlsx"
    )

    # ==================================================
    # SKIP IF ALREADY RUN TODAY
    # ==================================================

    if (
        HISTORY_FILE.exists()
        and excel_file.exists()
    ):
        print(
            f"Rejection log already exists for {today}."
        )
        print(
            "Skipping rejection log creation."
        )
        return

    # ==================================================
    # TODAY'S HISTORY RECORDS
    # ==================================================

    meters = pd.DataFrame(
        {
            "rejection_date": today,
            "asset_type": "meter",
            "asset_id": invalid_meters[
                "meter_number"
            ].astype("string"),
        }
    )

    transformers = pd.DataFrame(
        {
            "rejection_date": today,
            "asset_type": "transformer",
            "asset_id": invalid_transformers[
                "transformer_floc"
            ].astype("string"),
        }
    )

    today_history = pd.concat(
        [
            meters,
            transformers,
        ],
        ignore_index=True,
    )

    today_history = today_history.drop_duplicates(
        subset=[
            "rejection_date",
            "asset_type",
            "asset_id",
        ]
    )

    # ==================================================
    # UPDATE HISTORY
    # ==================================================

    if HISTORY_FILE.exists():

        history = pd.read_parquet(
            HISTORY_FILE
        )

        history = history[
            history["rejection_date"]
            != today
        ]

        history = pd.concat(
            [
                history,
                today_history,
            ],
            ignore_index=True,
        )

    else:

        history = today_history

    atomic_parquet(history, HISTORY_FILE)

    # ==================================================
    # WRITE EXCEL
    # ==================================================

    temporary = REJECTED_DIR / f".w{uuid4().hex[:12]}.xlsx"
    try:
        with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
            invalid_meters.to_excel(writer, sheet_name="Meters", index=False)
            invalid_transformers.to_excel(writer, sheet_name="Transformers", index=False)
        os.replace(temporary, excel_file)
    finally:
        temporary.unlink(missing_ok=True)

    # ==================================================
    # SUMMARY
    # ==================================================

    print(
        f"{today}: "
        f"{len(meters):,} meters rejected | "
        f"{len(transformers):,} transformers rejected"
    )

    print(f"History: {HISTORY_FILE}")
    print(f"Today's Excel: {excel_file}")