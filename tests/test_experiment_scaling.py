import pandas as pd

import timeflow_ts as tfts


def test_experiment_scales_only_value_columns(tmp_path):
    train = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "x": [1.0, -9999.0, 3.0, 4.0],
            "y": [2.0, 2.0, -9999.0, 2.0],
        }
    )
    val = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02", periods=2, freq="h").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "x": [10.0, 11.0],
            "y": [20.0, 21.0],
        }
    )

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)

    exp = tfts.TimeFlowExperiment(
        time_column="timestamp",
        value_columns=None,
        sentinel_value=-9999.0,
        window=2,
        fill_missing=False,
        fill_defaults=True,
        fill_outliers=False,
        latin=True,
        time_format="%Y-%m-%d %H:%M:%S",
    )

    scaler_path = tmp_path / "scaler.pkl"
    result = exp.run(
        train_files=[str(train_path)],
        val_files=[str(val_path)],
        test_files=None,
        fit_scaler=True,
        scaler_path=str(scaler_path),
        show_plots=False,
    )

    assert "timestamp" in result.train_df.columns
    assert "timestamp" in result.val_df.columns

    # Ensure timestamp values are unchanged (not standardized)
    assert result.train_df["timestamp"].dtype.kind in ("M", "m")

    # Scaled columns should not contain the sentinel.
    assert not (result.train_df[["x", "y"]] == -9999.0).any().any()
    assert not (result.val_df[["x", "y"]] == -9999.0).any().any()

    assert scaler_path.exists()

