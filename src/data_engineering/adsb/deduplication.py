"""Shared helpers for merging duplicate canonical ADS-B observations."""

import math
from collections.abc import Sequence

from data_engineering.adsb.parquet_schema import COLUMNS


DEDUPE_KEY_COLUMNS = ("time", "icao")
DEDUPE_SCORE_EXCLUDE_COLUMNS = set(DEDUPE_KEY_COLUMNS)
DEDUPE_PRIORITY_WEIGHTS = {
    # Prefer rows with real aircraft state over rows that only have metadata.
    "lat": 8,
    "lon": 8,
    "baro_altitude_ft": 8,
    "geom_altitude_ft": 8,
    "baro_rate_fpm": 10,
    "geom_rate_fpm": 10,
    "ground_speed_kt": 8,
    "indicated_airspeed_kt": 6,
    "true_airspeed_kt": 6,
    "mach": 6,
    "track_deg": 8,
    "track_rate_deg_s": 6,
    "roll_deg": 6,
    "magnetic_heading_deg": 6,
    "true_heading_deg": 6,
    "flags": 4,
    "type": 4,
}
COLUMN_INDEX = {column: index for index, column in enumerate(COLUMNS)}


def has_information(value) -> bool:
    """Return whether a field contains useful information for merging."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def row_information_score(row: Sequence) -> int:
    """Score a canonical row, prioritizing aircraft state over metadata."""
    return sum(
        DEDUPE_PRIORITY_WEIGHTS.get(column, 1)
        for column, value in zip(COLUMNS, row)
        if column not in DEDUPE_SCORE_EXCLUDE_COLUMNS
        and has_information(value)
    )


def merge_duplicate_rows(rows: Sequence[Sequence]) -> tuple:
    """Combine rows with one ``(time, icao)`` into one best-available row.

    The most informative aircraft-state row wins conflicting scalar values.
    Empty values in that row are backfilled from lower-ranked rows. Input
    order provides deterministic tie-breaking.
    """
    if not rows:
        raise ValueError("At least one ADS-B row is required")
    if len(rows) == 1:
        return tuple(rows[0])

    key = tuple(rows[0][COLUMN_INDEX[column]] for column in DEDUPE_KEY_COLUMNS)
    if any(
        tuple(row[COLUMN_INDEX[column]] for column in DEDUPE_KEY_COLUMNS) != key
        for row in rows[1:]
    ):
        raise ValueError("Cannot merge ADS-B rows with different (time, icao) keys")

    ranked_rows = sorted(rows, key=row_information_score, reverse=True)
    merged = list(ranked_rows[0])
    for column_index, column in enumerate(COLUMNS):
        if column in DEDUPE_KEY_COLUMNS:
            continue
        for row in ranked_rows:
            value = row[column_index]
            if has_information(value):
                merged[column_index] = value
                break
    return tuple(merged)


def merge_duplicate_column_rows(
    cols: dict[str, list],
    row_indices: Sequence[int],
) -> dict:
    """Column-oriented adapter for :func:`merge_duplicate_rows`."""
    rows = [
        tuple(cols[column][row_index] for column in COLUMNS)
        for row_index in row_indices
    ]
    merged = merge_duplicate_rows(rows)
    return dict(zip(COLUMNS, merged))


def dedupe_cols_by_time_icao(cols: dict[str, list]) -> tuple[dict[str, list], int]:
    """Merge duplicate ``(time, icao)`` rows, preserving first-key order."""
    row_count = len(cols["time"])
    if row_count < 2:
        return cols, 0

    rows_by_key: dict[tuple, list[int]] = {}
    for row_index in range(row_count):
        key = tuple(cols[column][row_index] for column in DEDUPE_KEY_COLUMNS)
        rows_by_key.setdefault(key, []).append(row_index)

    duplicate_count = sum(len(row_indices) - 1 for row_indices in rows_by_key.values())
    if duplicate_count == 0:
        return cols, 0

    deduped_cols = {column: [] for column in COLUMNS}
    for row_indices in rows_by_key.values():
        if len(row_indices) == 1:
            merged = {
                column: cols[column][row_indices[0]]
                for column in COLUMNS
            }
        else:
            merged = merge_duplicate_column_rows(cols, row_indices)
        for column in COLUMNS:
            deduped_cols[column].append(merged[column])

    return deduped_cols, duplicate_count
