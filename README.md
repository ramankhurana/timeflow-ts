# timeflow-ts

`timeflow-ts` supports practical time-series workflows:
- preprocessing (missing timestamps, missing values, sentinel defaults, optional outliers)
- yearly cleaning and saving
- merging many cleaned yearly CSV files into one dataset
- fitting `StandardScaler` on train and transforming val/test with the same scaler

## Install

```python
pip install timeflow-ts
```

## Use as drop-in replacement

If your old code used:

```python
from utils.processing import TimeSeriesProcessor
```

replace with:

```python
from timeflow_ts import TimeSeriesProcessor
```

## 1) Preprocess yearly files (same style as your code)

```python
from timeflow_ts import TimeSeriesProcessor

def prepareYearlyData(year: str = ""):
    processor = TimeSeriesProcessor(time_column="Date Time")
    processor.latin = False

    # Optional toggles
    processor.fillmiss = True        # interpolate + ffill missing values
    processor.filloutlier = False    # keep False if you only want defaults + missing handling

    if year != "2024":
        files_to_clean = [
            f"dataset_raw/mpi_roof_{year}a.csv",
            f"dataset_raw/mpi_roof_{year}b.csv",
        ]
    else:
        files_to_clean = [f"dataset_raw/mpi_roof_{year}.csv"]

    cleaned_files = processor.process_multiple_files(files_to_clean, output_suffix="_cleaned", plot=True)
    return cleaned_files

prepareYearlyData("2016")
prepareYearlyData("2017")
```

You can also use the built-in convenience method:

```python
processor = TimeSeriesProcessor(time_column="Date Time", latin=False)
processor.fillmiss = True
processor.preprocess_year(year="2018", base_dir="dataset_raw", output_suffix="_cleaned", plot=True)
```

## 2) Merge many cleaned yearly files

```python
from timeflow_ts import TimeSeriesProcessor

years = ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
cleaned_files = [f"dataset_raw/mpi_roof_{iy}_cleaned.csv" for iy in years]

processor = TimeSeriesProcessor(time_column="Date Time", latin=False)
merged_df = processor.merge_files(
    cleaned_files,
    output_file="dataset_raw/merged_all_years_cleaned.csv",
    sort_by_time=True,
)

print(merged_df.shape)
```

## 3) Train/Val/Test scaling (fit on train, transform val/test)

This keeps your time column and scales only features.

```python
from timeflow_ts import TimeSeriesProcessor

train_years = ["2016", "2019", "2022"]
val_years = ["2017", "2020", "2023"]
test_years = ["2018", "2021", "2024"]

train_files = [f"dataset_raw/mpi_roof_{iy}_cleaned.csv" for iy in train_years]
val_files = [f"dataset_raw/mpi_roof_{iy}_cleaned.csv" for iy in val_years]
test_files = [f"dataset_raw/mpi_roof_{iy}_cleaned.csv" for iy in test_years]

processor = TimeSeriesProcessor(time_column="Date Time")
processor.latin = False

train_scaled, val_scaled, test_scaled = processor.process_train_val_test1(
    train_files, val_files, test_files, scaler_path="dataset_raw/scaler.pkl"
)

def save_csvfiles(df, cols_to_drop, outputfile):
    df.drop(columns=cols_to_drop, inplace=False).to_csv(outputfile, index=False)

cols_to_drop = ["Tpot (K)", "Tdew (degC)", "Tlog (degC)", "VPmax (mbar)", "rho (g/m**3)"]
save_csvfiles(train_scaled, cols_to_drop, "dataset_raw/train.csv")
save_csvfiles(val_scaled, cols_to_drop, "dataset_raw/val.csv")
save_csvfiles(test_scaled, cols_to_drop, "dataset_raw/test.csv")
```

## Notes

- `process_multiple_files(...)` does sanity checks + sentinel filling by default.
- set `processor.fillmiss = True` to fill missing values too.
- set `processor.filloutlier = True` to clip/replace outliers.

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

