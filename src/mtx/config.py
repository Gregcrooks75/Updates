"""Local path configuration for SmartLink."""

import os
from pathlib import Path

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configured_directory(
    environment_variable: str,
    default_path: Path,
) -> Path:
    """Return a resolved directory from an environment variable or default."""

    configured_path = os.getenv(environment_variable)
    path = Path(configured_path) if configured_path else default_path

    return path.expanduser().resolve()


# ---------------------------------------------------------------------
# Pipeline directories
# ---------------------------------------------------------------------

INTERPOLATED_VOLTAGE_DIRECTORY = configured_directory(
    "SMARTLINK_INTERPOLATED_VOLTAGE_DIRECTORY",
    PROJECT_ROOT / "data" / "intermediate" / "interpolated_voltages",
)

AMI_CACHE_DIRECTORY = configured_directory(
    "SMARTLINK_AMI_CACHE_DIRECTORY",
    PROJECT_ROOT / "data" / 'cache' / "ami"
)
    
WEEKLY_VOLTAGE_DIRECTORY = configured_directory(
    "SMARTLINK_WEEKLY_VOLTAGE_DIRECTORY",
    PROJECT_ROOT / "data" / "processed" / "weekly_voltages",
)

METER_PAIR_DIRECTORY = configured_directory(
    "SMARTLINK_METER_PAIR_DIRECTORY",
    PROJECT_ROOT / "data" / "processed" / "meter_pairs",
)

METER_PAIR_FEATURE_DIRECTORY = configured_directory(
    "SMARTLINK_METER_PAIR_FEATURE_DIRECTORY",
    PROJECT_ROOT / "data" / "processed" / "meter_pair_features",
)

SPATIAL_FEATURE_DIRECTORY = configured_directory(
    "SMARTLINK_SPATIAL_FEATURE_DIRECTORY",
    PROJECT_ROOT / "data" / "processed" / "spatial_features",
)

METER_PAIR_TILE_COLUMNS = {
    "left_id": "meter_1",
    "right_id": "meter_2",
    "left_lon": "meter_1_LONGITUDE",
    "left_lat": "meter_1_LATITUDE",
    "right_lon": "meter_2_LONGITUDE",
    "right_lat": "meter_2_LATITUDE",
}

METER_TRANSFORMER_TILE_COLUMNS = {
    "left_id": "meter_number",
    "right_id": "candidate_transformer",
    "left_lon": "LONGITUDE",
    "left_lat": "LATITUDE",
    "right_lon": "Location_X",
    "right_lat": "Location_Y",
}

OSM_LAYER_FILE = configured_directory( "SMARTLINK_OSM_LAYER_FILE", PROJECT_ROOT / "data" / "reference" / "osm" / "osm_layers.pkl")

# Keep this root short on Windows. Existing source caches retain their layout.
M2T_DIRECTORY = configured_directory(
    "SMARTLINK_M2T_DIRECTORY", PROJECT_ROOT / "data" / "processed" / "m2t"
)
M2T_GROUP_DIRECTORY = M2T_DIRECTORY / "groups"
M2T_CANDIDATE_DIRECTORY = M2T_DIRECTORY / "candidates"
M2T_CANDIDATE_HISTORY_DIRECTORY = M2T_DIRECTORY / "history"
M2T_GROUP_HISTORY_DIRECTORY = M2T_DIRECTORY / "group_history"
TRANSFORMER_CACHE_DIRECTORY = configured_directory(
    "SMARTLINK_TRANSFORMER_CACHE_DIRECTORY",
    PROJECT_ROOT / "data" / "cache" / "transformers",
)
ADAM_CACHE_DIRECTORY = configured_directory(
    "SMARTLINK_ADAM_CACHE_DIRECTORY", PROJECT_ROOT / "data" / "cache" / "adam"
)
