from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ScalerBundle:
    scaler_path: str | None
    scaler: StandardScaler


def fit_scaler(
    df: pd.DataFrame,
    *,
    value_columns: list[str],
    scaler_path: str | None = None,
) -> tuple[pd.DataFrame, ScalerBundle]:
    """
    Fit StandardScaler on `value_columns` and return scaled dataframe.
    """
    scaler = StandardScaler()
    scaled = df.copy()
    scaled[value_columns] = scaler.fit_transform(df[value_columns])

    if scaler_path:
        joblib.dump(scaler, scaler_path)

    return scaled, ScalerBundle(scaler_path=scaler_path, scaler=scaler)


def transform_with_scaler(
    df: pd.DataFrame,
    *,
    value_columns: list[str],
    scaler: StandardScaler | None = None,
    scaler_path: str | None = None,
) -> pd.DataFrame:
    """
    Transform `value_columns` using an existing fitted scaler.
    """
    if scaler is None:
        if not scaler_path:
            raise ValueError("Either `scaler` or `scaler_path` must be provided.")
        scaler = joblib.load(scaler_path)

    scaled = df.copy()
    scaled[value_columns] = scaler.transform(df[value_columns])
    return scaled

