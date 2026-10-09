"""Configuration for one SmartLink pipeline run."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from math import isfinite

from mtx.config import (
    METER_PAIR_DIRECTORY,
    WEEKLY_VOLTAGE_DIRECTORY,
)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Parameters shared across the SmartLink pipeline."""
    
    week_start: str
    week_end: str
    correlation_start: str
    correlation_end: str
    force_ami_refresh: bool = False
    force_transformer_refresh: bool = False
    force_itron_refresh: bool = False
    meter_correlation_radius: float = 550
    correlation_method: str = "spearman"

    weekly_partition_size: int = 25_000
    seed_batch_size: int = 5_000
    
    # --------------------------------------------------
    # Meter-to-transformer configuration
    # --------------------------------------------------
    transformer_candidate_radius_ft: float = 550.0
    # Source used to build meter groups
    # Options:
    # "itron"
    # "smartlink"
    group_source: str = "itron"
    # Division column in transformer reference data
    transformer_division_col: str = "DivName"
    overwrite_m2t_outputs: bool = False
    transformer_candidate_batch_size: int = 25_000

    partitioned_divisions: tuple[str, ...] = (
        "Central",
        "Lancaster",
    )

    ami_columns: tuple[str, ...] = (
        "Address",
        "ZipCode",
        "City",
        "Division",
        "LATITUDE",
        "LONGITUDE",
        "Circuit",
        "SuperiorFLOC",
        "ConnectionObject",
        "PremiseID",
    )

    similarity_columns: tuple[str, ...] = (
        "Address",
        "ConnectionObject",
        "PremiseID",
    )
    
    exact_match_columns: tuple[str, ...] = (
        "ZipCode",
        "City",
        "Circuit",
    )
    
    def __post_init__(self) -> None:
        """Validate configuration values."""

        week_start = date.fromisoformat(self.week_start)
        week_end = date.fromisoformat(self.week_end)

        correlation_start = date.fromisoformat(
            self.correlation_start
        )
        correlation_end = date.fromisoformat(
            self.correlation_end
        )

        if week_start > week_end:
            raise ValueError(
                "week_start must not exceed week_end."
            )

        if not (
            week_start
            <= correlation_start
            <= correlation_end
            <= week_end
        ):
            raise ValueError(
                "Correlation dates must fall inside the weekly range."
            )

        if self.meter_correlation_radius <= 0:
            raise ValueError(
                "meter_correlation_radius must be positive."
            )

        if self.weekly_partition_size <= 0:
            raise ValueError(
                "weekly_partition_size must be positive."
            )

        if self.seed_batch_size <= 0:
            raise ValueError(
                "seed_batch_size must be positive."
            )

        if not isfinite(self.transformer_candidate_radius_ft) or self.transformer_candidate_radius_ft <= 0:
            raise ValueError("transformer_candidate_radius_ft must be positive.")
        if self.transformer_candidate_batch_size <= 0:
            raise ValueError("transformer_candidate_batch_size must be positive.")
        if self.group_source not in {"itron", "smartlink"}:
            raise ValueError("group_source must be itron or smartlink.")

        if self.correlation_method not in {
            "pearson",
            "spearman",
        }:
            raise ValueError(
                "correlation_method must be pearson or spearman."
            )

        ami_columns = set(self.ami_columns)

        if not set(self.similarity_columns).issubset(
            ami_columns
        ):
            raise ValueError(
                "All similarity columns must appear in ami_columns."
            )

        if not set(self.exact_match_columns).issubset(
            ami_columns
        ):
            raise ValueError(
                "All exact-match columns must appear in ami_columns."
            )

    @property
    def week_label(self) -> str:
        """Return the weekly date label."""

        return (
            f"{self.week_start.replace('-', '')}_"
            f"{self.week_end.replace('-', '')}"
        )

    @property
    def correlation_label(self) -> str:
        """Return the correlation date label."""

        return (
            f"{self.correlation_start.replace('-', '')}_"
            f"{self.correlation_end.replace('-', '')}"
        )

    @property
    def weekly_directory(self) -> Path:
        """Return the parameter-specific weekly output directory."""

        return (
            WEEKLY_VOLTAGE_DIRECTORY
            / f"week_{self.week_label}"
            / f"chunk_{self.weekly_partition_size}"
        )
    @property
    def pair_directory(self) -> Path:
        """Return the parameter-specific pair directory."""

        radius = f"{self.meter_correlation_radius:g}"

        return (
            METER_PAIR_DIRECTORY
            / f"{self.week_label}"
            / f"corr{self.correlation_label}"
            / f"rd_{radius}ft"
            / self.correlation_method
            / f"batch_{self.seed_batch_size}"
        )
