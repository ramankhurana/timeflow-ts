# Usage

## Create an experiment

```python
import timeflow_ts as tfts

exp = tfts.TimeFlowExperiment(
    time_column="timestamp",
    value_columns=None,
    sentinel_value=-9999,
    window=5,
    fill_missing=False,
    fill_outliers=False,
    latin=True,
    time_format="%d.%m.%Y %H:%M:%S",
)
```

## Run preprocessing + scaling

```python
result = exp.run(
    train_files=["yearA.csv"],
    val_files=["yearB.csv"],
    test_files=None,
    fit_scaler=True,
    scaler_path="scaler.pkl",
    show_plots=False,
)

print(result.metrics["train"])
train_scaled = result.train_df
```

