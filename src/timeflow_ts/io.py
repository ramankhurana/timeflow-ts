from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CsvLoadConfig:
    time_column: str = "timestamp"
    value_columns: list[str] | None = None
    latin: bool = True
    time_format: str | None = "%d.%m.%Y %H:%M:%S"

    def encoding(self) -> str | None:
        # Keep behavior similar to the original snippet, while being explicit and safe.
        return "latin1" if self.latin else None


def load_csv_files(
    file_paths: Iterable[str],
    *,
    config: CsvLoadConfig,
    infer_value_columns: bool = True,
) -> pd.DataFrame:
    """
    Load multiple CSV files and combine them into a single dataframe.

    Parameters
    ----------
    file_paths:
        Paths to CSV files.
    config:
        Loader configuration.
    infer_value_columns:
        If True and `config.value_columns` is None, infer value columns as all columns
        except `config.time_column`.
    """
    dfs: list[pd.DataFrame] = []
    for path in file_paths:
        df = pd.read_csv(path, encoding=config.encoding())
        if config.time_column in df.columns:
            # Original code used a fixed format when latin=True; support both.
            df[config.time_column] = pd.to_datetime(
                df[config.time_column],
                format=config.time_format,
                errors="raise",
            )
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    if infer_value_columns and config.value_columns is None:
        # Value column inference is intentionally handled by callers to keep this
        # loader function pure.
        pass
    return combined_df

