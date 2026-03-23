# Contributing to timeflot-ts

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
- `src/timeflot_ts/preprocessing.py`
- `src/timeflot_ts/metrics.py`
- `src/timeflot_ts/diagnostics.py`

Then wire them into `TimeFlotExperiment` if they are part of the one-stop pipeline.

