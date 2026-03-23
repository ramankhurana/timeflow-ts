# timeflow-ts

`timeflow-ts` is a one-stop library for experimentation with time series data:
- load and sanity-check CSV time series
- fill missing values and sentinel defaults
- handle outliers
- plot diagnostics and raw series
- compute metrics
- benchmark preprocessing steps
- scale train/val/test consistently for ML workflows

## Quick start

```python
import timeflow_ts as tfts

exp = tfts.TimeFlowExperiment(
    time_column="timestamp",
    value_columns=None,          # inferred from loaded columns
    sentinel_value=-9999,
    window=5,
    fill_missing=False,
    fill_outliers=False,
    latin=True,
    time_format="%d.%m.%Y %H:%M:%S",
)

result = exp.run(
    train_files=["yearA.csv"],
    val_files=["yearB.csv"],
    test_files=None,
    fit_scaler=True,
    scaler_path="scaler.pkl",
)

print(result.metrics["train"])
train_scaled = result.train_df
```

## Docs

API reference is generated automatically via MkDocs + mkdocstrings.

## Publish to GitHub + PyPI

1. Initialize a git repo and push to GitHub (repo name should match your PyPI metadata).
2. Update `mkdocs.yml` (`site_url`/`repo_url`) and the `docs` workflow if needed.
3. Bump the version in `pyproject.toml`.
4. Build and upload:

```bash
python -m build
twine upload dist/*
```

