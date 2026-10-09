from pathlib import Path
from datetime import date
import os
import uuid

import pandas as pd
import polars as pl

from mtx.db.db_GIS import get_devices_with_structures
from mtx.config import TRANSFORMER_CACHE_DIRECTORY

CACHE_DIR = TRANSFORMER_CACHE_DIRECTORY
CACHE_FILE = CACHE_DIR / "ny_transformers.parquet"
METADATA_FILE = CACHE_DIR / "ny_transformers_metadata.txt"


## Cache validation to check if transformer data was checked today
def cache_is_current() -> bool:
    """
    Return True when the transformer cache was created today.
    """
    if not CACHE_FILE.exists():
        return False

    if not METADATA_FILE.exists():
        return False

    try:
        cached_date = METADATA_FILE.read_text().strip()
    except OSError:
        return False

    return cached_date == date.today().isoformat()

## Transformer Validation 

def validate_transformers(
    df: pd.DataFrame,
) -> None:
    """
    Basic validation before allowing transformer data
    to become the pipeline cache.
    """
    if df.empty:
        raise ValueError(
            "Transformer query returned no rows."
        )

    required_columns = {
        "Location_Y",
        "Location_X",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Transformer data missing required columns: "
            f"{missing}"
        )


## Atomic Save Cache

def save_transformer_cache(
    df: pd.DataFrame,
) -> None:
    """
    Atomically save transformer data so a failed write
    cannot destroy the existing cache.
    """
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = CACHE_DIR / (
        f".w{uuid.uuid4().hex[:12]}.tmp"
    )

    try:
        df.to_parquet(
            temp_file,
            index=False,
        )

        # Verify write succeeded.
        check_df = pl.read_parquet(
            temp_file
        )

        if check_df.height != len(df):
            raise ValueError(
                "Transformer cache validation failed: "
                f"expected {len(df)} rows, "
                f"found {check_df.height}."
            )

        os.replace(
            temp_file,
            CACHE_FILE,
        )

        metadata_temp = CACHE_DIR / f".m{uuid.uuid4().hex[:12]}.tmp"
        try:
            metadata_temp.write_text(date.today().isoformat(), encoding="utf-8")
            os.replace(metadata_temp, METADATA_FILE)
        finally:
            metadata_temp.unlink(missing_ok=True)

    finally:
        if temp_file.exists():
            temp_file.unlink()
            
## Load or Refresh

def load_or_refresh_ny_transformers(
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load today's cached NY transformer data when possible.
    Otherwise query NYGIS and replace the transformer cache.
    """
    if (
        not force_refresh
        and cache_is_current()
    ):
        print(
            f"[Transformers] Reusing cached data: "
            f"{CACHE_FILE}"
        )

        cached = pd.read_parquet(CACHE_FILE)
        validate_transformers(cached)
        return cached

    print(
        "[Transformers] Refreshing transformer data from NYGIS..."
    )

    transformer_df = get_devices_with_structures(
        server="nygisprd19",
        database="NYGIS",
        device="transformer",
    )
    # Protect against GIS returning Polars.
    if isinstance(
        transformer_df,
        pl.DataFrame,
    ):
        transformer_df = (
            transformer_df.to_pandas()
        )

    validate_transformers(
        transformer_df
    )

    save_transformer_cache(
        transformer_df
    )

    print(
        f"[Transformers] Cached "
        f"{len(transformer_df):,} transformers."
    )

    return transformer_df