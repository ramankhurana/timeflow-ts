from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def fill_missing(
    df: pd.DataFrame, *, value_columns: list[str], method: str = "linear"
) -> pd.DataFrame:
    """
    Fill missing values in `value_columns`.

    - First interpolate (default `linear`)
    - Then forward-fill remaining NaNs
    """
    df = df.copy()
    df[value_columns] = df[value_columns].interpolate(method=method)
    df[value_columns] = df[value_columns].fillna(method="ffill")
    return df


def fill_defaults(
    df: pd.DataFrame,
    *,
    value_columns: list[str],
    sentinel_value: float = -9999,
    window: int = 5,
) -> pd.DataFrame:
    """
    Replace sentinel defaults using rolling mean of past values.

    Steps per column:
    1) Replace sentinel values with NaN
    2) Compute rolling mean on past values only via `shift(1)`
    3) Fill NaNs with that rolling mean
    4) If still NaN (start of series), forward-fill
    """
    df = df.copy()
    for col in value_columns:
        df[col] = df[col].replace(sentinel_value, np.nan)
        rolling_mean = df[col].shift(1).rolling(window=window, min_periods=1).mean()
        df[col] = df[col].fillna(rolling_mean)
        df[col] = df[col].ffill()
    return df


def fill_outliers(
    df: pd.DataFrame,
    *,
    value_columns: list[str],
    method: Literal["clip", "nan"] = "clip",
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> pd.DataFrame:
    """
    Handle outliers using a simple IQR-style rule based on quantiles.

    If `method="clip"`, values are clipped to bounds.
    If `method="nan"`, outliers are replaced with NaN.
    """
    df = df.copy()
    for col in value_columns:
        q1 = df[col].quantile(lower_quantile)
        q3 = df[col].quantile(upper_quantile)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        if method == "clip":
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == "nan":
            mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            df.loc[mask, col] = np.nan
        else:
            raise ValueError(f"Unsupported method: {method}")

    return df

