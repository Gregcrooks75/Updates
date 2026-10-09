"""Reuse today's ADAM extract; validate labels before atomic cache publication."""
from __future__ import annotations
from datetime import date, datetime
import pandas as pd
from mtx.config import ADAM_CACHE_DIRECTORY
from mtx.db.db_AMI import get_adam_validated_data
from mtx.utils.daily_cache import atomic_json, atomic_parquet, file_digest, read_json

CACHE_DIR = ADAM_CACHE_DIRECTORY
CACHE_FILE = CACHE_DIR / "adam_validated.parquet"
METADATA_FILE = CACHE_DIR / "adam_validated_metadata.json"


def cache_date():
    metadata = read_json(METADATA_FILE)
    if metadata:
        try:
            return datetime.fromisoformat(metadata["created_at"]).date()
        except (KeyError, TypeError, ValueError):
            pass
    # Preserve the existing metadata-free cache compatibility.
    if CACHE_FILE.exists():
        return datetime.fromtimestamp(CACHE_FILE.stat().st_mtime).date()
    return None


def validate_adam(frame):
    missing = {"meter_number", "Correct_Transformer_FLOC"} - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"ADAM extract is empty or missing columns: {sorted(missing)}")
    frame = frame.copy()
    for column in ("meter_number", "Correct_Transformer_FLOC"):
        clean = frame[column].astype("string").str.strip()
        frame[column] = clean.mask(clean.str.lower().isin({"", "nan", "none", "null", "<na>"}))
    frame = frame.drop_duplicates().reset_index(drop=True)
    if frame["meter_number"].isna().any() or frame["meter_number"].duplicated().any():
        raise ValueError("ADAM has duplicate/conflicting meter labels or missing meter keys; resolve them before caching/history.")
    return frame


def write_metadata(row_count):
    atomic_json({"created_at": datetime.now().isoformat(), "row_count": row_count,
                 "sha256": file_digest(CACHE_FILE)}, METADATA_FILE)


def load_or_refresh_adam_data(force_refresh=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = read_json(METADATA_FILE)
    if not force_refresh and CACHE_FILE.exists() and cache_date() == date.today():
        if metadata and metadata.get("sha256") and metadata["sha256"] != file_digest(CACHE_FILE):
            raise ValueError("Today's ADAM cache failed its checksum; refresh it before history commits.")
        frame = validate_adam(pd.read_parquet(CACHE_FILE))
        if metadata and metadata.get("row_count") != len(frame):
            raise ValueError("Today's ADAM cache row count differs from its metadata.")
        if not metadata or "sha256" not in metadata:
            write_metadata(len(frame))
        return frame
    frame = validate_adam(get_adam_validated_data())
    # Invalidate the old completion marker before replacing the data.
    METADATA_FILE.unlink(missing_ok=True)
    atomic_parquet(frame, CACHE_FILE)
    write_metadata(len(frame))
    return frame
