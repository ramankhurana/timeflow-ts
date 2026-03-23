from __future__ import annotations

import pandas as pd

from .diagnostics import compute_time_gap_stats


def compute_metrics(
    df: pd.DataFrame,
    *,
    time_column: str,
    value_columns: list[str],
    sentinel_value: float | None = -9999,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    exclude_columns: list[str] | None = None,
) -> dict:
    """
    Compute simple experiment metrics for a time series dataframe.

    Metrics included:
    - time_gap_min/max/median
    - missing_total
    - sentinel_default_total (if sentinel_value provided)
    - outlier_total via quantile-based IQR bounds
    """
    df_sorted = df.copy()
    df_sorted[time_column] = pd.to_datetime(df_sorted[time_column])
    df_sorted = df_sorted.sort_values(by=time_column).reset_index(drop=True)

    gaps = compute_time_gap_stats(df_sorted, time_column=time_column)

    missing_total = int(df_sorted[value_columns].isna().sum().sum())

    if exclude_columns is None:
        exclude_columns = []

    outlier_total = 0
    for col in value_columns:
        if col in exclude_columns or col == time_column:
            continue
        q1 = df_sorted[col].quantile(lower_quantile)
        q3 = df_sorted[col].quantile(upper_quantile)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_total += int(
            ((df_sorted[col] < lower_bound) | (df_sorted[col] > upper_bound)).sum()
        )

    sentinel_total = None
    if sentinel_value is not None:
        sentinel_total = 0
        for col in value_columns:
            if col in exclude_columns or col == time_column:
                continue
            sentinel_total += int((df_sorted[col] == sentinel_value).sum())

    return {
        "time_gap_min": gaps.min,
        "time_gap_max": gaps.max,
        "time_gap_median": gaps.median,
        "missing_total": missing_total,
        "sentinel_default_total": sentinel_total,
        "outlier_total": outlier_total,
    }

