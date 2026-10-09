"""Cache normalized ITRON transformer subgroup assignments."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from mtx.utils.daily_cache import file_digest


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "itron"

CACHE_FILE = CACHE_DIR / "itron_tx_subgroups.parquet"

METADATA_FILE = (
    CACHE_DIR / "itron_tx_subgroups_metadata.json"
)


DEFAULT_REQUIRED_COLUMNS = {
    "meter_number",
    "Tx_Subgroup_ID",
    "tx_group_id",
    "tx_group_size",
    "group_source",
    "group_type",
}


def cache_is_fresh(
    max_age_days: int = 1,
    *,
    cache_file: Path = CACHE_FILE,
    metadata_file: Path = METADATA_FILE,
    cache_day=None,
) -> bool:
    """Use calendar-day reuse when cache_day is given; retain rolling-age callers."""

    if (
        max_age_days < 0
        or not cache_file.is_file()
        or not metadata_file.is_file()
    ):
        return False

    try:
        metadata = json.loads(
            metadata_file.read_text(encoding="utf-8")
        )

        created_at = datetime.fromisoformat(
            metadata["created_at"]
        )

        now = (
            datetime.now(created_at.tzinfo)
            if created_at.tzinfo is not None
            else datetime.now()
        )

        age = now - created_at
        if age < timedelta(0):
            return False
        if cache_day is not None:
            saved_day = metadata.get("cache_date") or created_at.astimezone().date().isoformat()
            if saved_day != str(cache_day)[:10]:
                return False
        return age <= timedelta(days=max_age_days)

    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def to_pandas(data: Any) -> pd.DataFrame:
    """Normalize pandas or Polars input to a defensive pandas copy."""

    if isinstance(data, pd.DataFrame):
        return data.copy()

    try:
        import polars as pl
    except ImportError:
        pl = None

    if pl is not None and isinstance(data, pl.DataFrame):
        return data.to_pandas()

    raise TypeError(
        "ITRON grouping builder must return a pandas or Polars "
        f"DataFrame, not {type(data).__name__}."
    )


def validate_itron_groups(
    df: pd.DataFrame,
    required_columns: Iterable[str] | None = DEFAULT_REQUIRED_COLUMNS,
) -> None:
    """Reject unusable ITRON group data before caching or reuse."""

    if df.empty:
        raise ValueError(
            "ITRON grouping returned zero rows. "
            "Existing cache was preserved."
        )

    required = set(required_columns or ())

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "ITRON grouping is missing required columns: "
            f"{sorted(missing)}"
        )

    if "meter_number" in df.columns:

        meter = (
            df["meter_number"]
            .astype("string")
            .str.strip()
        )

        invalid_meter = (
            meter.isna()
            | meter.eq("")
        )

        if invalid_meter.any():
            raise ValueError(
                f"ITRON grouping contains "
                f"{int(invalid_meter.sum()):,} "
                "invalid meter identifiers."
            )

        duplicates = meter.duplicated(keep=False)

        if duplicates.any():
            raise ValueError(
                f"ITRON grouping contains "
                f"{int(duplicates.sum()):,} rows "
                "whose meter identifiers are duplicated."
            )

    if "tx_group_id" in df.columns:

        group_id = (
            df["tx_group_id"]
            .astype("string")
            .str.strip()
        )

        if (group_id.isna() | group_id.eq("")).any():
            raise ValueError(
                "ITRON grouping contains invalid "
                "tx_group_id values."
            )

    if "tx_group_size" in df.columns:

        sizes = pd.to_numeric(
            df["tx_group_size"],
            errors="coerce",
        )

        if sizes.isna().any() or sizes.lt(1).any():
            raise ValueError(
                "ITRON grouping contains invalid "
                "tx_group_size values."
            )


def write_cache_atomically(
    df: pd.DataFrame,
    *,
    cache_file: Path = CACHE_FILE,
    metadata_file: Path = METADATA_FILE,
    required_columns: Iterable[str] | None = DEFAULT_REQUIRED_COLUMNS,
) -> None:
    """Validate, write, read back, and atomically replace cache files."""

    validate_itron_groups(
        df,
        required_columns=required_columns,
    )

    cache_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    token = uuid.uuid4().hex[:12]

    temporary_cache = cache_file.with_name(
        f".w{token}.tmp"
    )

    temporary_metadata = metadata_file.with_name(
        f".m{token}.tmp"
    )

    try:

        df.to_parquet(
            temporary_cache,
            index=False,
        )

        verification = pd.read_parquet(
            temporary_cache
        )

        validate_itron_groups(
            verification,
            required_columns=required_columns,
        )

        if len(verification) != len(df):
            raise ValueError(
                "ITRON cache verification failed: expected "
                f"{len(df):,} rows and read back "
                f"{len(verification):,}."
            )

        metadata = {
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "cache_date": date.today().isoformat(),
            "sha256": file_digest(temporary_cache),
            "rows": len(df),
            "columns": list(df.columns),
        }

        temporary_metadata.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        os.replace(
            temporary_cache,
            cache_file,
        )

        os.replace(
            temporary_metadata,
            metadata_file,
        )

    finally:

        temporary_cache.unlink(
            missing_ok=True
        )

        temporary_metadata.unlink(
            missing_ok=True
        )


def load_or_refresh_itron_groups(
    refresh_fn: Callable[[], Any],
    *,
    force_refresh: bool = False,
    max_age_days: int = 1,
    cache_day=None,
    required_columns: Iterable[str] | None = DEFAULT_REQUIRED_COLUMNS,
    cache_file: Path = CACHE_FILE,
    metadata_file: Path = METADATA_FILE,
) -> pd.DataFrame:
    """Reuse a valid cache or rebuild it with ``refresh_fn``."""

    if (
        not force_refresh
        and cache_is_fresh(
            max_age_days=max_age_days,
            cache_day=cache_day,
            cache_file=cache_file,
            metadata_file=metadata_file,
        )
    ):
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            if metadata.get("sha256") and metadata["sha256"] != file_digest(cache_file):
                raise ValueError("ITRON cache checksum failed.")

            cached = pd.read_parquet(
                cache_file
            )

            validate_itron_groups(
                cached,
                required_columns=required_columns,
            )

            print(
                f"[ITRON] Reused {len(cached):,} "
                "cached subgroup rows."
            )

            return cached

        except Exception as exc:
            print(
                f"[ITRON] Cache is unreadable "
                f"or invalid: {exc}"
            )

    if force_refresh:
        print(
            "[ITRON] Forced refresh requested."
        )

    elif not cache_file.exists():
        print(
            "[ITRON] No cache found."
        )

    else:
        print(
            "[ITRON] Cache is stale or invalid."
        )

    print(
        "[ITRON] Rebuilding transformer subgroups..."
    )

    refreshed = to_pandas(
        refresh_fn()
    )

    validate_itron_groups(
        refreshed,
        required_columns=required_columns,
    )

    write_cache_atomically(
        refreshed,
        cache_file=cache_file,
        metadata_file=metadata_file,
        required_columns=required_columns,
    )

    print(
        f"[ITRON] Cached "
        f"{len(refreshed):,} subgroup rows."
    )

    print(
        f"[ITRON] Cache file: {cache_file}"
    )

    return refreshed