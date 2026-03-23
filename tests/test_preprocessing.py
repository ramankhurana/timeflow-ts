import pandas as pd

from timeflot_ts.preprocessing import fill_defaults


def test_fill_defaults_replaces_sentinel_with_past_rolling_mean():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="h"),
            "a": [1.0, -9999.0, 3.0, -9999.0, 5.0],
        }
    )

    out = fill_defaults(df, value_columns=["a"], sentinel_value=-9999.0, window=2)

    # At index 1, past values within window=[0] => mean=1.0
    assert out.loc[1, "a"] == 1.0

    # Implementation uses `shift(1)` before rolling, so for index 3 it averages past raw values
    # within the window: previous value is index 2 => mean=3.0 (index 1 is NaN at roll time).
    assert out.loc[3, "a"] == 3.0

