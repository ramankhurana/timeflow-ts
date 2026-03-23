from __future__ import annotations

import pandas as pd

from .diagnostics import sanity_checks
from .io import CsvLoadConfig, load_csv_files
from .preprocessing import fill_defaults, fill_missing, fill_outliers
from .scaling import fit_scaler, transform_with_scaler


class TimeSeriesProcessor:
    """
    Backwards-compatible processor inspired by the original snippet.

    Prefer `TimeFlotExperiment` for new projects.
    """

    def __init__(
        self,
        time_column: str = "timestamp",
        value_columns: list[str] | None = None,
        latin: bool = True,
    ):
        """
        Parameters
        ----------
        time_column:
            Name of timestamp column.
        value_columns:
            Columns to process; if None they will be inferred after loading.
        latin:
            If True, load CSV files with `latin1` encoding; otherwise use pandas default.
        """
        self.time_column = time_column
        self.value_columns = value_columns
        self.scaler = None
        self.filloutlier = False
        self.fillmiss = False
        self.latin = latin

    def load_files(
        self, file_paths: list[str], *, time_format: str | None = "%d.%m.%Y %H:%M:%S"
    ) -> pd.DataFrame:
        config = CsvLoadConfig(
            time_column=self.time_column,
            value_columns=self.value_columns,
            latin=self.latin,
            time_format=time_format,
        )
        df = load_csv_files(file_paths, config=config, infer_value_columns=False)

        if self.value_columns is None:
            self.value_columns = [col for col in df.columns if col != self.time_column]
        return df

    def sanity_checks(self, df: pd.DataFrame, *, plot: bool = True) -> pd.DataFrame:
        if self.value_columns is None:
            raise ValueError("value_columns must be set (or inferred by load_files).")
        df_sorted, _ = sanity_checks(
            df,
            time_column=self.time_column,
            value_columns=self.value_columns,
            sentinel_value=-9999,
            plot=plot,
        )
        return df_sorted

    def fill_missing(self, df: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
        if self.value_columns is None:
            raise ValueError("value_columns must be set (or inferred by load_files).")
        return fill_missing(df, value_columns=self.value_columns, method=method)

    def fill_defaults(
        self, df: pd.DataFrame, sentinel_value: float = -9999, window: int = 5
    ) -> pd.DataFrame:
        if self.value_columns is None:
            raise ValueError("value_columns must be set (or inferred by load_files).")
        return fill_defaults(
            df, value_columns=self.value_columns, sentinel_value=sentinel_value, window=window
        )

    def fill_outliers(self, df: pd.DataFrame, method: str = "clip") -> pd.DataFrame:
        if self.value_columns is None:
            raise ValueError("value_columns must be set (or inferred by load_files).")
        return fill_outliers(df, value_columns=self.value_columns, method=method)  # type: ignore[arg-type]

    def save_clean_csv(self, df: pd.DataFrame, original_path: str, suffix: str = "_cleaned") -> str:
        if "a.csv" in original_path or "b.csv" in original_path:
            new_path = (
                original_path.replace("a.csv", f"{suffix}.csv").replace("b.csv", f"{suffix}.csv")
            )
        else:
            new_path = original_path.replace(".csv", f"{suffix}.csv")
        df.to_csv(new_path, index=False)
        return new_path

    def fit_scaler(self, df: pd.DataFrame, *, scaler_path: str = "scaler.pkl") -> pd.DataFrame:
        if self.value_columns is None:
            raise ValueError("value_columns must be set (or inferred by load_files).")
        df_scaled, bundle = fit_scaler(
            df, value_columns=self.value_columns, scaler_path=scaler_path
        )
        self.scaler = bundle.scaler
        return df_scaled

    def transform_with_scaler(
        self, df: pd.DataFrame, *, scaler_path: str = "scaler.pkl"
    ) -> pd.DataFrame:
        if self.value_columns is None:
            raise ValueError("value_columns must be set (or inferred by load_files).")
        df_scaled = transform_with_scaler(
            df,
            value_columns=self.value_columns,
            scaler=self.scaler,
            scaler_path=scaler_path,
        )
        return df_scaled

    def process_multiple_files(
        self,
        input_files: list[str],
        *,
        output_suffix: str = "_cleaned",
        plot: bool = True,
    ) -> list[str]:
        df = self.load_files(input_files)
        df = self.sanity_checks(df, plot=plot)

        if self.fillmiss:
            df = self.fill_missing(df)

        df = self.fill_defaults(df)

        if self.filloutlier:
            df = self.fill_outliers(df)

        file = input_files[0]
        cleaned_file = self.save_clean_csv(df, file, suffix=output_suffix)
        return [cleaned_file]

    def process_train_val_test(
        self,
        train_files: list[str],
        val_files: list[str] | None = None,
        test_files: list[str] | None = None,
        *,
        scaler_path: str = "scaler.pkl",
    ):
        train_df = self.load_files(train_files)
        train_scaled = self.fit_scaler(train_df, scaler_path=scaler_path)

        val_scaled = None
        test_scaled = None

        if val_files:
            val_df = self.load_files(val_files)
            val_scaled = self.transform_with_scaler(val_df, scaler_path=scaler_path)

        if test_files:
            test_df = self.load_files(test_files)
            test_scaled = self.transform_with_scaler(test_df, scaler_path=scaler_path)

        return train_scaled, val_scaled, test_scaled

