from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from .diagnostics import sanity_checks
from .io import CsvLoadConfig, load_csv_files
from .metrics import compute_metrics
from .preprocessing import fill_defaults, fill_missing, fill_outliers
from .scaling import fit_scaler as fit_scaler_bundle
from .scaling import transform_with_scaler


@dataclass(frozen=True)
class TimeFlowExperimentConfig:
    time_column: str = "timestamp"
    value_columns: list[str] | None = None
    latin: bool = True
    time_format: str | None = "%d.%m.%Y %H:%M:%S"

    # Preprocessing controls
    sentinel_value: float = -9999
    window: int = 5
    fill_missing: bool = False
    fill_defaults: bool = True
    fill_outliers: bool = False
    outlier_method: str = "clip"  # "clip" or "nan"
    missing_method: str = "linear"

    # Quantile-based outlier detection
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95


@dataclass
class TimeFlowResult:
    train_df: pd.DataFrame
    val_df: pd.DataFrame | None = None
    test_df: pd.DataFrame | None = None

    # Metrics before/after preprocessing (raw + cleaned)
    metrics: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    scaler_path: str | None = None


class TimeFlowExperiment:
    """
    One-stop API for time series experimentation.

    It loads CSVs, runs sanity checks, optionally preprocesses (missing/sentinel/outliers),
    computes metrics, plots, benchmarks the run, and scales train/val/test consistently.
    """

    def __init__(self, **kwargs):
        self.config = TimeFlowExperimentConfig(**kwargs)

    def _load_split(self, file_paths: list[str]) -> pd.DataFrame:
        load_config = CsvLoadConfig(
            time_column=self.config.time_column,
            value_columns=self.config.value_columns,
            latin=self.config.latin,
            time_format=self.config.time_format,
        )
        return load_csv_files(file_paths, config=load_config, infer_value_columns=False)

    def _infer_value_columns(self, df: pd.DataFrame) -> list[str]:
        if self.config.value_columns is not None:
            return list(self.config.value_columns)
        if self.config.time_column not in df.columns:
            raise ValueError(
                f"Expected time column {self.config.time_column!r} in dataframe columns."
            )
        return [c for c in df.columns if c != self.config.time_column]

    def _preprocess(self, df: pd.DataFrame, *, value_columns: list[str]) -> pd.DataFrame:
        if self.config.fill_missing:
            df = fill_missing(df, value_columns=value_columns, method=self.config.missing_method)
        if self.config.fill_defaults:
            df = fill_defaults(
                df,
                value_columns=value_columns,
                sentinel_value=self.config.sentinel_value,
                window=self.config.window,
            )
        if self.config.fill_outliers:
            df = fill_outliers(
                df,
                value_columns=value_columns,
                method=self.config.outlier_method,  # type: ignore[arg-type]
                lower_quantile=self.config.lower_quantile,
                upper_quantile=self.config.upper_quantile,
            )
        return df

    def _apply_pipeline_to_split(
        self,
        *,
        file_paths: list[str],
        value_columns: list[str],
        show_plots: bool,
        plot_kwargs: dict | None,
    ) -> tuple[pd.DataFrame, dict, dict, dict]:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        df_raw = self._load_split(file_paths)
        timings["load_seconds"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        df_sorted, checks = sanity_checks(
            df_raw,
            time_column=self.config.time_column,
            value_columns=value_columns,
            sentinel_value=self.config.sentinel_value,
            plot=show_plots,
            plot_kwargs=plot_kwargs,
        )
        timings["sanity_checks_seconds"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        metrics_before = compute_metrics(
            df_sorted,
            time_column=self.config.time_column,
            value_columns=value_columns,
            sentinel_value=self.config.sentinel_value,
            lower_quantile=self.config.lower_quantile,
            upper_quantile=self.config.upper_quantile,
        )
        timings["metrics_before_seconds"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        df_clean = self._preprocess(df_sorted, value_columns=value_columns)
        timings["preprocess_seconds"] = time.perf_counter() - t3

        t4 = time.perf_counter()
        df_clean_sorted, checks_after = sanity_checks(
            df_clean,
            time_column=self.config.time_column,
            value_columns=value_columns,
            sentinel_value=self.config.sentinel_value,
            plot=show_plots,
            plot_kwargs=plot_kwargs,
        )
        timings["sanity_checks_after_seconds"] = time.perf_counter() - t4

        t5 = time.perf_counter()
        metrics_after = compute_metrics(
            df_clean_sorted,
            time_column=self.config.time_column,
            value_columns=value_columns,
            sentinel_value=self.config.sentinel_value,
            lower_quantile=self.config.lower_quantile,
            upper_quantile=self.config.upper_quantile,
        )
        timings["metrics_after_seconds"] = time.perf_counter() - t5

        metrics = {"before": metrics_before, "after": metrics_after}
        checks_dict = {"before": checks, "after": checks_after}
        return df_clean_sorted, metrics, checks_dict, timings

    def run(
        self,
        *,
        train_files: list[str],
        val_files: list[str] | None = None,
        test_files: list[str] | None = None,
        fit_scaler: bool = True,
        scaler_path: str | None = "scaler.pkl",
        show_plots: bool = False,
        plot_kwargs: dict | None = None,
    ) -> TimeFlowResult:
        """
        Run the full pipeline for train/val/test.
        """
        if plot_kwargs is None:
            plot_kwargs = {}

        timings: dict[str, dict[str, float]] = {}
        metrics: dict[str, dict] = {}
        checks: dict[str, dict] = {}

        # Train first to infer columns and (optionally) fit scaler
        df_train_clean, train_metrics, train_checks, train_timings = self._apply_pipeline_to_split(
            file_paths=train_files,
            value_columns=self._infer_value_columns(self._load_split(train_files)),
            show_plots=show_plots,
            plot_kwargs=plot_kwargs,
        )
        value_columns = self._infer_value_columns(df_train_clean)

        # Re-run with inferred columns consistently
        df_train_clean, train_metrics, train_checks, train_timings = self._apply_pipeline_to_split(
            file_paths=train_files,
            value_columns=value_columns,
            show_plots=show_plots,
            plot_kwargs=plot_kwargs,
        )
        timings["train"] = train_timings
        metrics["train"] = train_metrics
        checks["train"] = train_checks

        scaler_used = None
        df_train_scaled = df_train_clean
        if fit_scaler:
            df_train_scaled, scaler_bundle = fit_scaler_bundle(
                df_train_clean,
                value_columns=value_columns,
                scaler_path=scaler_path,
            )
            scaler_used = scaler_bundle

        df_val_scaled = None
        if val_files:
            df_val_clean, val_metrics, val_checks, val_timings = self._apply_pipeline_to_split(
                file_paths=val_files,
                value_columns=value_columns,
                show_plots=show_plots,
                plot_kwargs=plot_kwargs,
            )
            timings["val"] = val_timings
            metrics["val"] = val_metrics
            checks["val"] = val_checks

            if fit_scaler:
                df_val_scaled = transform_with_scaler(
                    df_val_clean,
                    value_columns=value_columns,
                    scaler=scaler_used.scaler if scaler_used else None,
                    scaler_path=scaler_path,
                )
            else:
                df_val_scaled = df_val_clean

        df_test_scaled = None
        if test_files:
            df_test_clean, test_metrics, test_checks, test_timings = self._apply_pipeline_to_split(
                file_paths=test_files,
                value_columns=value_columns,
                show_plots=show_plots,
                plot_kwargs=plot_kwargs,
            )
            timings["test"] = test_timings
            metrics["test"] = test_metrics
            checks["test"] = test_checks

            if fit_scaler:
                df_test_scaled = transform_with_scaler(
                    df_test_clean,
                    value_columns=value_columns,
                    scaler=scaler_used.scaler if scaler_used else None,
                    scaler_path=scaler_path,
                )
            else:
                df_test_scaled = df_test_clean

        return TimeFlowResult(
            train_df=df_train_scaled,
            val_df=df_val_scaled,
            test_df=df_test_scaled,
            metrics=metrics,
            checks=checks,
            timings=timings,
            scaler_path=scaler_path if fit_scaler else None,
        )

