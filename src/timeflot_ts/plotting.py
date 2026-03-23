from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_timeseries(
    df: pd.DataFrame,
    *,
    time_column: str,
    value_columns: list[str],
    subplots: bool = True,
    figsize: tuple[int, int] = (20, 60),
    show: bool = True,
    title: str | None = None,
):
    """Plot raw timeseries for `value_columns`."""
    fig = df.plot(
        x=time_column,
        y=value_columns,
        subplots=subplots,
        figsize=figsize,
        title=title,
    )
    if show:
        plt.show()
    return fig

