from __future__ import annotations
from pathlib import Path

DIVISION_MAPPING = {
    "Central/Rochester": "Central",
    "Lakeshore/Sodus": "Sodus",
    "Genesee Valley/Filmore": "Fillmore",
}


VALID_DIVISIONS = {
    "Auburn",
    "Binghamton",
    "Brewster",
    "Canandaigua",
    "Central",
    "Elmira",
    "Fillmore",
    "Geneva",
    "Hornell",
    "Ithaca",
    "Lancaster",
    "Liberty",
    "Lockport",
    "Mechanicville",
    "Oneonta",
    "Plattsburgh",
    "Sodus",
}



def save_standardized_cache(
    ami_df,
    transformer_df,
    itron_groups_df,
    ami_file,
    transformer_file,
    itron_file,
):
    ami_df.to_parquet(
        ami_file,
        index=False,
    )

    transformer_df.to_parquet(
        transformer_file,
        index=False,
    )

    itron_groups_df.to_parquet(
        itron_file,
        index=False,
    )

    print(
        "Standardized cache files updated."
    )

def standardize_divisions(
    df,
    name,
):
    df = df.copy()

    df["Division"] = (
        df["Division"]
        .astype("string")
        .str.strip()
        .str.title()
        .replace(DIVISION_MAPPING)
    )

    # Remove rows without a division
    df = df[
        df["Division"].notna()
    ].copy()

    unexpected = (
        set(df["Division"].unique())
        - VALID_DIVISIONS
    )

    if unexpected:
        raise ValueError(
            f"{name} has unexpected divisions: "
            f"{sorted(unexpected)}"
        )

    print(
        f"{name}: "
        f"{df['Division'].nunique()} divisions"
    )

    return df


def normalize_identifiers(frame, columns):
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            cleaned = frame[column].astype("string").str.strip()
            frame[column] = cleaned.mask(cleaned.str.lower().isin({"", "nan", "none", "null", "<na>"}))
    return frame


def standardize_all_inputs(
    ami_df,
    transformer_df,
    itron_groups_df,
    division_gdf,
    adam_df
):

    # ==================================================
    # AMI
    # ==================================================

    ami_df = ami_df.rename(
        columns={
            "MeterID": "meter_number",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
            "SuperiorFLOC": "current_meter_floc",
        }
    )

    # ==================================================
    # TRANSFORMERS
    # ==================================================

    transformer_df = transformer_df.rename(
        columns={
            "FuncLoc": "transformer_floc",
            'SAPID':'sap_id',
            'GISID':'gis_id',
            "Location_X": "longitude",
            "Location_Y": "latitude",
            "Transformer_kVA": "transformer_kva",
            "DivName": "Division",
        }
    )

    # Remove vertical structures
    transformer_df = transformer_df[
        transformer_df["StructureType"] != "Vert"
    ].copy()

    # ==================================================
    # ITRON
    # ==================================================

    itron_groups_df = itron_groups_df.rename(
        columns={
            "division": "Division",
        }
    )

    # ==================================================
    # DIVISION POLYGONS
    # ==================================================

    # division_gdf already contains:
    #
    # Division
    # geometry

    ami_df = normalize_identifiers(ami_df, ["meter_number", "current_meter_floc"])
    transformer_df = normalize_identifiers(transformer_df, ["transformer_floc"])
    itron_groups_df = normalize_identifiers(itron_groups_df, ["meter_number", "Tx_Subgroup_ID", "tx_group_id", "current_meter_floc"])
    adam_df = normalize_identifiers(adam_df, ["meter_number", "Correct_Transformer_FLOC", "MeterFLOC"])
    if ami_df["meter_number"].isna().any() or ami_df["meter_number"].duplicated().any():
        raise ValueError("AMI meter_number must be unique and nonblank after normalization.")

    # ==================================================
    # STANDARDIZE DIVISIONS
    # ==================================================

    ami_df = standardize_divisions(
        ami_df,
        "AMI",
    )

    transformer_df = standardize_divisions(
        transformer_df,
        "Transformers",
    )

    itron_groups_df = standardize_divisions(
        itron_groups_df,
        "ITRON Subgroups",
    )

    division_gdf = standardize_divisions(
        division_gdf,
        "Division Polygons",
    )
    
    adam_df = standardize_divisions(
        adam_df,
        "Adam's Data",
    )

    return (
        ami_df,
        transformer_df,
        itron_groups_df,
        division_gdf,
        adam_df
    )