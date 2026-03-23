from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .plotting import plot_timeseries


@dataclass(frozen=True)
class TimeGapStats:
    min: pd.Timedelta
    max: pd.Timedelta
    median: pd.Timedelta


def compute_time_gap_stats(df: pd.DataFrame, *, time_column: str) -> TimeGapStats:
    """Compute basic time-delta statistics for a sorted time series."""
    time_diff = df[time_column].diff().dropna()
    return TimeGapStats(min=time_diff.min(), max=time_diff.max(), median=time_diff.median())


@dataclass(frozen=True)
class SanityChecks:
    time_gap_stats: TimeGapStats
    missing_summary: pd.Series
    outliers_per_column: pd.DataFrame
    sentinel_default_counts: pd.DataFrame | None = None


def sanity_checks(
    df: pd.DataFrame,
    *,
    time_column: str,
    value_columns: list[str],
    sentinel_value: float = -9999,
    plot: bool = False,
    plot_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, SanityChecks]:
    """
    Perform checks for missing timestamps, sorting, missing values, outliers, and sentinel defaults.

    Returns
    -------
    (sorted_df, checks)
    """
    if plot_kwargs is None:
        plot_kwargs = {}

    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(by=time_column).reset_index(drop=True)

    gap_stats = compute_time_gap_stats(df, time_column=time_column)

    missing_summary = df[value_columns].isna().sum()

    outliers: dict[str, int] = {}
    for col in value_columns:
        q1 = df[col].quantile(0.05)
        q3 = df[col].quantile(0.95)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers[col] = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())

    outliers_df = pd.DataFrame.from_dict(outliers, orient="index", columns=["outlier_count"])

    sentinel_counts: pd.DataFrame | None = None
    if sentinel_value is not None:
        sentinel_counts_dict: dict[str, int] = {}
        for col in value_columns:
            if col == time_column:
                continue
            sentinel_counts_dict[col] = int((df[col] == sentinel_value).values.sum())
        sentinel_counts = pd.DataFrame.from_dict(
            sentinel_counts_dict, orient="index", columns=["sentinel_default_count"]
        )

    checks = SanityChecks(
        time_gap_stats=gap_stats,
        missing_summary=missing_summary,
        outliers_per_column=outliers_df,
        sentinel_default_counts=sentinel_counts,
    )

    if plot:
        plot_timeseries(
            df,
            time_column=time_column,
            value_columns=value_columns,
            **plot_kwargs,
        )

    return df, checks

