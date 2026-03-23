# Contributing to timeflow-ts

## Development setup

1. Create a virtual environment (recommended).
2. Install dev dependencies:

```bash
pip install -e ".[dev]"
```

3. Run tests:

```bash
pytest -q
```

4. Lint:

```bash
ruff check .
```

## Code style

Follow `ruff` (configured in `pyproject.toml`). Keep functions small and prefer typed public APIs.

## Adding new preprocessing/metrics

Prefer adding new functions under:
- `src/timeflow_ts/preprocessing.py`
- `src/timeflow_ts/metrics.py`
- `src/timeflow_ts/diagnostics.py`

Then wire them into `TimeFlowExperiment` if they are part of the one-stop pipeline.

