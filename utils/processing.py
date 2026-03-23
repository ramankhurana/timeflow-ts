"""
Compatibility shim for legacy imports.

Existing code can keep:
    from utils.processing import TimeSeriesProcessor
"""

from timeflow_ts import TimeSeriesProcessor

__all__ = ["TimeSeriesProcessor"]

