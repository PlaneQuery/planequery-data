"""Noise filtering for ADS-B trajectories before flight segmentation.

This is adapted from Traffic's default ``FilterAboveSigmaMedian`` filter. For
each tracked feature, it compares the raw values with a rolling median and
rejects values whose squared deviation is greater than the rolling median
error threshold. Rejected and originally missing values are then backward
filled followed by forward filling, matching ``Flight.filter()``'s default
strategy.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy import signal


# Traffic column names translated to the normalized PlaneQuery ADS-B schema.
# Latitude and longitude use Traffic's default numeric-feature kernel of 17.
DEFAULT_FILTER_KERNELS: dict[str, int | tuple[int, ...]] = {
    "lat": 17,
    "lon": 17,
    "baro_altitude_ft": (17, 53),
    "geom_altitude_ft": (17, 53),
    "ground_speed_kt": 5,
    "track_deg": 17,
}


def _as_kernel_sequence(kernels: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(kernels, int):
        kernels = (kernels,)
    else:
        kernels = tuple(kernels)

    if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
        raise ValueError("median filter kernels must be positive odd integers")
    return kernels


def _backward_then_forward_fill(values: np.ndarray) -> np.ndarray:
    """Fill NaNs with the next value, then use the previous value at the end."""
    filled = values.copy()
    indices = np.arange(filled.size)

    missing = np.isnan(filled)
    next_indices = np.where(missing, filled.size, indices)
    next_indices = np.minimum.accumulate(next_indices[::-1])[::-1]
    has_next = missing & (next_indices < filled.size)
    filled[has_next] = filled[next_indices[has_next]]

    missing = np.isnan(filled)
    previous_indices = np.where(missing, -1, indices)
    previous_indices = np.maximum.accumulate(previous_indices)
    has_previous = missing & (previous_indices >= 0)
    filled[has_previous] = filled[previous_indices[has_previous]]

    return filled


def _filter_feature(
    values: np.ndarray,
    kernels: int | Sequence[int],
) -> np.ndarray:
    filtered = values.astype(float, copy=True)

    for kernel in _as_kernel_sequence(kernels):
        # This matches Traffic's behavior: a cascade is skipped when there are
        # not enough messages for its median window.
        if kernel > filtered.size:
            continue

        median = signal.medfilt(filtered, kernel)
        squared_error = (filtered - median) ** 2
        sigma = np.sqrt(signal.medfilt(squared_error, kernel))
        filtered[squared_error > sigma] = np.nan

    return _backward_then_forward_fill(filtered)


def _filtered_trajectory(
    trajectory: pl.DataFrame,
    column_kernels: dict[str, int | tuple[int, ...]],
) -> pl.DataFrame:
    expressions: list[pl.Series] = []

    for column, kernels in column_kernels.items():
        if column not in trajectory.columns:
            continue

        original_dtype = trajectory.schema[column]
        values = trajectory.get_column(column).to_numpy().astype(float)
        filtered = _filter_feature(values, kernels)
        expressions.append(
            pl.Series(column, filtered)
            .fill_nan(None)
            .cast(original_dtype, strict=False)
            .alias(column)
        )

    if not expressions:
        return trajectory
    return trajectory.with_columns(expressions)


def filter_adsb_messages(
    df_adsb: pl.DataFrame,
    *,
    column_kernels: dict[str, int | tuple[int, ...]] | None = None,
) -> pl.DataFrame:
    """Apply the default trajectory filter independently to each ICAO.

    Rows are sorted by ICAO and timestamp before filtering. The input schema is
    preserved, and ADS-B columns unknown to the filter pass through unchanged.
    """
    required_columns = {"icao", "time"}
    missing_columns = required_columns.difference(df_adsb.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"ADS-B filtering requires columns: {missing}")

    sorted_df = df_adsb.sort(["icao", "time"])
    if sorted_df.is_empty():
        return sorted_df

    kernels = (
        DEFAULT_FILTER_KERNELS
        if column_kernels is None
        else column_kernels
    )
    trajectories = sorted_df.partition_by(
        "icao",
        maintain_order=True,
        include_key=True,
    )
    return pl.concat(
        [
            _filtered_trajectory(trajectory, kernels)
            for trajectory in trajectories
        ]
    )
