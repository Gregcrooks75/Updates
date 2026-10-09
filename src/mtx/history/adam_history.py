"""
Daily Adam training-data history using the same layout as AMI history.

Paste into a notebook cell or import from this file. All Adam constants and
private helpers have their own names so they can coexist with the AMI code.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import polars as pl


# ============================================================
# PROJECT PATH AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ADAM_HISTORY_DIR = PROJECT_ROOT / "data" / "cache" / "adam" / "history"

ADAM_KEY = "meter_number"
ADAM_LABEL = "Correct_Transformer_FLOC"


ADAM_SUMMARY_SCHEMA = {
    "date": pl.Date,
    "meter_count": pl.Int64,
    "added": pl.Int64,
    "removed": pl.Int64,
    "modified": pl.Int64,
    "labeled_meter_count": pl.Int64,
    "new_training_examples": pl.Int64,
    "corrected_labels": pl.Int64,
    "lost_training_examples": pl.Int64,
    "metadata_only_changes": pl.Int64,
    "training_changed": pl.Boolean,
}


ADAM_CHANGE_SCHEMA = {
    "change_date": pl.String,
    ADAM_KEY: pl.String,
    "change_type": pl.String,
    "field": pl.String,
    "previous_value": pl.String,
    "current_value": pl.String,
    "run_id": pl.String,
    "changed_at": pl.String,
}


ADAM_TRAINING_SCHEMA = {
    "change_date": pl.String,
    ADAM_KEY: pl.String,
    "event": pl.String,
    "previous_label": pl.String,
    "current_label": pl.String,
    "run_id": pl.String,
    "changed_at": pl.String,
}


# Used only when reading older history.
# Never used to standardize current_df.
ADAM_OLD_NAMES = {
    "MeterID": ADAM_KEY,
    "MeterId": ADAM_KEY,
    "MeterNumber": ADAM_KEY,
    "Meter_Number": ADAM_KEY,
    "division": "Division",
}


# ============================================================
# PRIVATE HELPERS
# ============================================================

def adamoldnames(frame):
    """Rename legacy Adam columns when reading older history."""

    names = (
        frame.collect_schema().names()
        if isinstance(frame, pl.LazyFrame)
        else frame.columns
    )

    renames = {
        old: new
        for old, new in ADAM_OLD_NAMES.items()
        if old in names and new not in names
    }

    if len(set(renames.values())) != len(renames):
        raise ValueError(
            "Multiple legacy Adam columns map to the same name."
        )

    return frame.rename(renames)


def adamsave(frame, path):
    """Finish a temporary Parquet file before replacing its destination."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    frame.write_parquet(
        temporary,
        compression="zstd",
    )

    temporary.replace(path)


def adamlabel(column):
    """
    Return a nonblank, stripped label expression for training comparisons.

    The original label text is retained in the full rows and field-level log.
    Label presence does not establish FLOC validity or model eligibility.
    """

    value = (
        pl.col(column)
        .cast(pl.String)
        .str.strip_chars()
    )

    return (
        pl.when(
            value.is_not_null()
            & value.ne("")
        )
        .then(value)
        .otherwise(None)
    )


def adamprepare(frame, description):
    """Validate and prepare an Adam DataFrame for history tracking."""

    missing = {
        ADAM_KEY,
        ADAM_LABEL,
    } - set(frame.columns)

    if missing:
        raise KeyError(
            f"{description} is missing required columns: "
            f"{sorted(missing)}"
        )

    reserved = {
        "change_date",
        "change_type",
        "row_version",
        "run_id",
        "changed_at",
    }

    conflicts = reserved.intersection(frame.columns)

    conflicts.update(
        column
        for column in frame.columns
        if column.endswith("__old")
    )

    if conflicts:
        raise ValueError(
            f"{description} contains reserved history columns: "
            f"{sorted(conflicts)}"
        )

    # Avoid converting floating-point missing IDs or labels
    # to the literal string "NaN".
    frame = (
        frame.with_columns(
            [
                (
                    pl.col(column).fill_nan(None)
                    if frame.schema[column].is_float()
                    else pl.col(column)
                )
                .cast(pl.String)
                .alias(column)
                for column in (
                    ADAM_KEY,
                    ADAM_LABEL,
                )
            ]
        )
        .with_columns(
            pl.col(ADAM_KEY).str.strip_chars()
        )
    )

    key = frame[ADAM_KEY]

    if (
        key.null_count()
        or key.eq("").any()
        or key.n_unique() != frame.height
    ):
        raise ValueError(
            f"{description}: '{ADAM_KEY}' must be unique, "
            "nonblank, and non-null. Resolve duplicate "
            "meters or labels before tracking."
        )

    return frame


def adamread_summary(root):
    """Read and validate the committed Adam daily summary."""

    path = Path(root) / "adam_daily_summary.parquet"

    if not path.exists():
        return pl.DataFrame(
            schema=ADAM_SUMMARY_SCHEMA
        )

    summary = pl.read_parquet(path)

    if (
        "date" not in summary.columns
        and "snapshot_date" in summary.columns
    ):
        summary = summary.rename(
            {"snapshot_date": "date"}
        )

    required = {
        "date",
        "meter_count",
        "added",
        "removed",
        "modified",
    }

    if not required.issubset(summary.columns):
        missing = required - set(summary.columns)

        raise ValueError(
            f"Adam summary is missing columns: "
            f"{sorted(missing)}"
        )

    summary = summary.with_columns(
        pl.col("date")
        .cast(pl.String)
        .str.slice(0, 10)
        .str.to_date()
    )

    if (
        summary["date"].null_count()
        or summary["date"].n_unique() != summary.height
    ):
        raise ValueError(
            "Adam summary dates must be non-null and unique."
        )

    # Older summaries may lack training counters.
    # Unknown past counts remain null.
    summary = summary.with_columns(
        [
            pl.lit(
                None,
                dtype=dtype,
            ).alias(column)
            for column, dtype in ADAM_SUMMARY_SCHEMA.items()
            if column not in summary.columns
        ]
    )

    return (
        summary
        .select(list(ADAM_SUMMARY_SCHEMA))
        .cast(ADAM_SUMMARY_SCHEMA)
        .sort("date")
    )


# ============================================================
# TRACK ADAM HISTORY
# ============================================================

def track_adam_history(
    current_df,
    snapshot_date=None,
    *,
    history_dir=ADAM_HISTORY_DIR,
):
    """
    Record one complete, standardized Adam extract per date.

    Parameters
    ----------
    current_df
        Pandas or Polars DataFrame containing the standardized Adam data.

    snapshot_date
        Date assigned to the snapshot. Defaults to today's date.

    history_dir
        Directory used to store the daily Adam history.

    Required columns
    ----------------
    meter_number
    Correct_Transformer_FLOC

    Notes
    -----
    All other columns, including Division, Correct, ValidationSource, and
    ValidationDate when present, are also compared and retained.

    First run:
        Stores complete INITIAL rows. All change counters are zero.

    Later dates:
        Compares the current data with previous_adam_snapshot.parquet.

    Same date:
        Returns the saved field changes without processing newer same-day
        edits.

    Returns
    -------
    pl.DataFrame
        Field-level Adam changes, using the same structure as AMI history.

    Important
    ---------
    Run one writer at a time. Dates must advance.

    Empty extracts, invalid keys, column changes, and incompatible types are
    rejected before new data is saved.

    Use a separate history_dir when testing with samples because using a
    subset against production history would imply that missing rows were
    removed.

    training_changed describes changes in nonblank meter-label pairs since
    the previous recorded date. It is not a comparison with the last trained
    model, a check of every model feature, or an automatic retraining
    decision.

    ValidationDate is source data. changed_at is the UTC observation time.

    This daily layout is separate from the earlier training_history/runs
    layout. Old per-run history is not automatically migrated or overwritten.
    """
    day = date.fromisoformat(
        str(
            snapshot_date
            or date.today()
        )[:10]
    )

    root = Path(history_dir)

    previous_file = (
        root
        / "previous_adam_snapshot.parquet"
    )

    summary_file = (
        root
        / "adam_daily_summary.parquet"
    )

    pending_file = (
        root
        / f"pending_adam_{day}.parquet"
    )

    daily_files = {
        folder: (
            root
            / folder
            / f"{day}.parquet"
        )
        for folder in (
            "changes",
            "changed_rows",
            "training_changes",
        )
    }

    # Refuse to mix the old per-run layout into this daily
    # history directory.
    if (root / "runs").exists():
        raise ValueError(
            "This directory contains older per-run Adam history. "
            "Use a separate directory for the daily layout, or "
            "migrate it first."
        )

    summary = adamread_summary(root)

    # Recover a run that committed its summary but did not
    # promote its snapshot.
    for pending in sorted(
        root.glob("pending_adam_*.parquet")
    ):
        pending_day = date.fromisoformat(
            pending.stem.removeprefix(
                "pending_adam_"
            )
        )

        if pending_day in summary["date"]:
            if pending_day != summary["date"].max():
                raise ValueError(
                    f"Unexpected stale pending file: {pending}"
                )

            for folder in daily_files:
                path = (
                    root
                    / folder
                    / f"{pending_day}.parquet"
                )

                if not path.exists():
                    raise FileNotFoundError(
                        "Committed Adam history is incomplete: "
                        f"{path}"
                    )

            pending.replace(previous_file)

        elif pending_day != day:
            raise ValueError(
                f"Finish the interrupted {pending_day} "
                f"Adam run first: {pending}"
            )

    first_run = not previous_file.exists()

    if first_run and summary.height:
        raise FileNotFoundError(
            "Adam summary exists but comparison snapshot "
            f"is missing: {previous_file}"
        )

    if (
        not first_run
        and summary.is_empty()
    ):
        raise ValueError(
            "Adam comparison snapshot exists but its daily "
            "summary is missing or empty."
        )

    # Return the already committed result for the same date.
    if day in summary["date"]:
        for path in daily_files.values():
            if not path.exists():
                raise FileNotFoundError(
                    "Saved Adam history is incomplete: "
                    f"{path}"
                )

        print(
            f"Adam history already processed for {day}; "
            "returning saved changes."
        )

        return pl.read_parquet(
            daily_files["changes"]
        )

    if (
        summary.height
        and day <= summary["date"].max()
    ):
        raise ValueError(
            "Use a date after the last recorded date: "
            f"{summary['date'].max()}."
        )

    if (
        any(
            path.exists()
            for path in daily_files.values()
        )
        and not pending_file.exists()
    ):
        raise ValueError(
            f"Adam history files exist for {day}, but its "
            "summary entry is missing."
        )

    # Convert Pandas input to Polars, or clone an existing
    # Polars DataFrame.
    current = (
        current_df.clone()
        if isinstance(current_df, pl.DataFrame)
        else pl.from_pandas(current_df)
    )

    if current.is_empty():
        raise ValueError(
            "The current Adam DataFrame is empty. "
            "History was not changed."
        )

    current = adamprepare(
        current,
        "Current standardized Adam data",
    )

    if first_run:
        old = current.head(0)

    else:
        old = adamprepare(
            adamoldnames(
                pl.read_parquet(previous_file)
            ),
            "Previous Adam snapshot",
        )

    # Compare every source column.
    # Permit an all-null column to acquire a type.
    if set(old.columns) != set(current.columns):
        raise ValueError(
            "Align standardized columns before tracking. "
            f"Only in previous: "
            f"{sorted(set(old.columns) - set(current.columns))}; "
            f"only in current: "
            f"{sorted(set(current.columns) - set(old.columns))}."
        )

    old = old.select(current.columns)

    old = old.cast(
        {
            column: current.schema[column]
            for column in old.columns
            if old.schema[column] == pl.Null
        }
    )

    current = current.cast(
        {
            column: old.schema[column]
            for column in current.columns
            if current.schema[column] == pl.Null
        }
    )

    if old.schema != current.schema:
        raise ValueError(
            "Column types changed; align input types before tracking. "
            f"Previous: {old.schema}. "
            f"Current: {current.schema}."
        )

    # --------------------------------------------------------
    # IDENTIFY ADDED, REMOVED, AND MODIFIED METERS
    # --------------------------------------------------------

    added = current.join(
        old.select(ADAM_KEY),
        on=ADAM_KEY,
        how="anti",
    )

    removed = old.join(
        current.select(ADAM_KEY),
        on=ADAM_KEY,
        how="anti",
    )

    joined = current.join(
        old,
        on=ADAM_KEY,
        how="inner",
        suffix="__old",
    )

    fields = [
        column
        for column in current.columns
        if column != ADAM_KEY
    ]

    different = {}

    for column in fields:
        new = pl.col(column)
        before = pl.col(
            column + "__old"
        )

        if current.schema[column].is_float():
            new = new.fill_nan(None)
            before = before.fill_nan(None)

        different[column] = (
            ~new.eq_missing(before)
        )

    modified = joined.filter(
        pl.any_horizontal(
            list(different.values())
        )
    )

    modified_ids = modified.select(
        ADAM_KEY
    )

    stamp = {
        "change_date": str(day),
        "run_id": uuid4().hex[:12],
        "changed_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    metadata = {
        key: pl.lit(value)
        for key, value in stamp.items()
    }

    null = pl.lit(
        None,
        dtype=pl.String,
    )

    # --------------------------------------------------------
    # FIELD-LEVEL CHANGE LOG
    # --------------------------------------------------------

    logs = [
        pl.DataFrame(
            schema=ADAM_CHANGE_SCHEMA
        )
    ]

    if not first_run:
        for frame, kind in (
            (added, "ADDED"),
            (removed, "REMOVED"),
        ):
            logs.append(
                frame.select(
                    ADAM_KEY,
                    pl.lit(kind).alias(
                        "change_type"
                    ),
                    null.alias("field"),
                    null.alias(
                        "previous_value"
                    ),
                    null.alias(
                        "current_value"
                    ),
                )
            )

        for field in fields:
            logs.append(
                modified.filter(
                    different[field]
                ).select(
                    ADAM_KEY,
                    pl.lit(
                        "MODIFIED"
                    ).alias(
                        "change_type"
                    ),
                    pl.lit(field).alias(
                        "field"
                    ),
                    pl.col(
                        field + "__old"
                    )
                    .cast(pl.String)
                    .alias(
                        "previous_value"
                    ),
                    pl.col(field)
                    .cast(pl.String)
                    .alias(
                        "current_value"
                    ),
                )
            )

    changes = (
        pl.concat(
            logs,
            how="diagonal_relaxed",
        )
        .with_columns(**metadata)
        .select(
            list(ADAM_CHANGE_SCHEMA)
        )
        .cast(ADAM_CHANGE_SCHEMA)
    )

    # --------------------------------------------------------
    # TRAINING-LABEL CHANGES
    # --------------------------------------------------------

    training_parts = [
        pl.DataFrame(
            schema=ADAM_TRAINING_SCHEMA
        )
    ]

    if not first_run:
        training_parts.append(
            added.filter(
                adamlabel(
                    ADAM_LABEL
                ).is_not_null()
            ).select(
                ADAM_KEY,
                pl.lit(
                    "ADDED_LABELED"
                ).alias("event"),
                null.alias(
                    "previous_label"
                ),
                adamlabel(
                    ADAM_LABEL
                ).alias(
                    "current_label"
                ),
            )
        )

        training_parts.append(
            removed.filter(
                adamlabel(
                    ADAM_LABEL
                ).is_not_null()
            ).select(
                ADAM_KEY,
                pl.lit(
                    "LABELED_ROW_REMOVED"
                ).alias("event"),
                adamlabel(
                    ADAM_LABEL
                ).alias(
                    "previous_label"
                ),
                null.alias(
                    "current_label"
                ),
            )
        )

        before = adamlabel(
            ADAM_LABEL + "__old"
        )

        after = adamlabel(
            ADAM_LABEL
        )

        event = (
            pl.when(
                before.is_null()
                & after.is_not_null()
            )
            .then(
                pl.lit(
                    "NEWLY_LABELED"
                )
            )
            .when(
                before.is_not_null()
                & after.is_null()
            )
            .then(
                pl.lit(
                    "LABEL_REMOVED"
                )
            )
            .otherwise(
                pl.lit(
                    "LABEL_CORRECTED"
                )
            )
        )

        training_parts.append(
            joined.filter(
                ~before.eq_missing(after)
            ).select(
                ADAM_KEY,
                event.alias("event"),
                before.alias(
                    "previous_label"
                ),
                after.alias(
                    "current_label"
                ),
            )
        )

    training_changes = (
        pl.concat(
            training_parts,
            how="diagonal_relaxed",
        )
        .with_columns(**metadata)
        .select(
            list(ADAM_TRAINING_SCHEMA)
        )
        .cast(ADAM_TRAINING_SCHEMA)
    )

    # --------------------------------------------------------
    # SAVE COMPLETE CHANGED ROW VERSIONS
    # --------------------------------------------------------

    if first_run:
        versions = [
            (
                current,
                "INITIAL",
                "CURRENT",
            )
        ]

    else:
        versions = [
            (
                added,
                "ADDED",
                "CURRENT",
            ),
            (
                removed,
                "REMOVED",
                "PREVIOUS",
            ),
            (
                old.join(
                    modified_ids,
                    on=ADAM_KEY,
                    how="semi",
                ),
                "MODIFIED_BEFORE",
                "PREVIOUS",
            ),
            (
                current.join(
                    modified_ids,
                    on=ADAM_KEY,
                    how="semi",
                ),
                "MODIFIED_AFTER",
                "CURRENT",
            ),
        ]

    rows = pl.concat(
        [
            frame.with_columns(
                **metadata,
                change_type=pl.lit(kind),
                row_version=pl.lit(version),
            )
            for frame, kind, version in versions
        ],
        how="diagonal_relaxed",
    )

    # --------------------------------------------------------
    # DAILY SUMMARY COUNTERS
    # --------------------------------------------------------

    def count_events(*events):
        return training_changes.filter(
            pl.col("event").is_in(events)
        ).height

    new_training = count_events(
        "ADDED_LABELED",
        "NEWLY_LABELED",
    )

    corrected = count_events(
        "LABEL_CORRECTED"
    )

    lost_training = count_events(
        "LABEL_REMOVED",
        "LABELED_ROW_REMOVED",
    )

    # Includes cosmetic label edits, such as surrounding
    # whitespace-only changes.
    metadata_only = modified_ids.join(
        training_changes.select(
            ADAM_KEY
        ),
        on=ADAM_KEY,
        how="anti",
    ).height

    daily = pl.DataFrame(
        [
            {
                "date": day,
                "meter_count": current.height,
                "added": (
                    0
                    if first_run
                    else added.height
                ),
                "removed": removed.height,
                "modified": modified.height,
                "labeled_meter_count": (
                    current.filter(
                        adamlabel(
                            ADAM_LABEL
                        ).is_not_null()
                    ).height
                ),
                "new_training_examples": (
                    new_training
                ),
                "corrected_labels": corrected,
                "lost_training_examples": (
                    lost_training
                ),
                "metadata_only_changes": (
                    metadata_only
                ),
                "training_changed": bool(
                    new_training
                    or corrected
                    or lost_training
                ),
            }
        ],
        schema=ADAM_SUMMARY_SCHEMA,
    )

    updated_summary = pl.concat(
        [
            summary,
            daily,
        ]
    )

    # The summary is the commit marker.
    # Update the comparison snapshot last.
    #
    # An interrupted, uncommitted same-date run is recomputed
    # using current_df.
    adamsave(
        current,
        pending_file,
    )

    adamsave(
        changes,
        daily_files["changes"],
    )

    adamsave(
        rows,
        daily_files["changed_rows"],
    )

    adamsave(
        training_changes,
        daily_files["training_changes"],
    )

    adamsave(
        updated_summary,
        summary_file,
    )

    pending_file.replace(
        previous_file
    )

    result = daily.row(
        0,
        named=True,
    )

    initial_message = (
        " - initial baseline"
        if first_run
        else ""
    )

    print(
        f"\nAdam history completed for {day}"
        f"{initial_message}"
    )

    print(
        f"Meters: {current.height:,} | "
        f"Labeled: "
        f"{result['labeled_meter_count']:,}"
    )

    print(
        f"Added: {result['added']:,} | "
        f"Removed: {result['removed']:,} | "
        f"Modified: {result['modified']:,}"
    )

    print(
        f"New training examples: "
        f"{new_training:,} | "
        f"Corrected labels: "
        f"{corrected:,} | "
        f"Lost examples: "
        f"{lost_training:,}"
    )

    print(
        "Metadata/formatting-only changes: "
        f"{metadata_only:,}"
    )

    print(
        "Training labels changed: "
        f"{result['training_changed']}"
    )

    print(
        f"History: {root}"
    )

    return changes


# ============================================================
# RETRIEVE SAVED HISTORY
# ============================================================

def get_adam_meter_history(
    meter_number,
    *,
    full_rows=False,
    training_only=False,
    history_dir=ADAM_HISTORY_DIR,
):
    """
    Return the saved history for one meter.

    By default, field-level changes are returned.

    Parameters
    ----------
    meter_number
        Meter number to retrieve.

    full_rows
        Return complete changed-row versions.

    training_only
        Return only training-label events.

    history_dir
        Adam daily history directory.

    Notes
    -----
    Only dates committed in the summary are included.
    Incomplete runs remain hidden.

    INITIAL rows are present only when full_rows=True.
    """

    if full_rows and training_only:
        raise ValueError(
            "Choose either full_rows=True or "
            "training_only=True."
        )

    if full_rows:
        folder = "changed_rows"

    elif training_only:
        folder = "training_changes"

    else:
        folder = "changes"

    root = Path(history_dir)
    queries = []

    for day in adamread_summary(root)["date"]:
        path = (
            root
            / folder
            / f"{day}.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                "Saved Adam history is incomplete: "
                f"{path}"
            )

        query = adamoldnames(
            pl.scan_parquet(path)
        )

        if (
            ADAM_KEY
            not in query.collect_schema().names()
        ):
            raise KeyError(
                f"Saved history is missing "
                f"'{ADAM_KEY}': {path}"
            )

        query = (
            query
            .with_columns(
                pl.col(ADAM_KEY)
                .cast(pl.String)
                .str.strip_chars()
            )
            .filter(
                pl.col(ADAM_KEY)
                == str(
                    meter_number
                ).strip()
            )
        )

        queries.append(query)

    if not queries:
        if full_rows:
            return pl.DataFrame()

        schema = (
            ADAM_TRAINING_SCHEMA
            if training_only
            else ADAM_CHANGE_SCHEMA
        )

        return pl.DataFrame(
            schema=schema
        )

    return (
        pl.concat(
            queries,
            how="diagonal_relaxed",
        )
        .collect()
        .sort(
            "change_date",
            maintain_order=True,
        )
    )


def get_adam_training_changes(
    snapshot_date=None,
    *,
    history_dir=ADAM_HISTORY_DIR,
):
    """
    Return training-label events for a committed date.

    The latest committed date is used by default.
    """

    root = Path(history_dir)
    summary = adamread_summary(root)

    if summary.is_empty():
        return pl.DataFrame(
            schema=ADAM_TRAINING_SCHEMA
        )

    day = (
        summary["date"].max()
        if snapshot_date is None
        else date.fromisoformat(
            str(snapshot_date)[:10]
        )
    )

    if day not in summary["date"]:
        raise ValueError(
            f"No committed Adam history for {day}."
        )

    path = (
        root
        / "training_changes"
        / f"{day}.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Saved Adam training history is missing: "
            f"{path}"
        )

    return pl.read_parquet(path)

