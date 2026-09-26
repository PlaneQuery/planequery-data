from enum import StrEnum

import polars as pl


class AircraftAdsbCategory(StrEnum):
    A1 = "A1"  # Light aircraft
    A2 = "A2"  # Small aircraft
    A3 = "A3"  # Large aircraft
    A4 = "A4"  # High-vortex large aircraft
    A5 = "A5"  # Heavy aircraft
    A6 = "A6"  # High-performance aircraft
    A7 = "A7"  # Rotorcraft
    B1 = "B1"  # Glider / sailplane
    B2 = "B2"  # Lighter-than-air
    B3 = "B3"  # Parachutist / skydiver
    B4 = "B4"  # Ultralight / hang-glider / paraglider
    B5 = "B5"  # Reserved
    B6 = "B6"  # Unmanned aerial vehicle
    B7 = "B7"  # Space / trans-atmospheric vehicle

LIKELY_COMMERCIAL_CATEGORIES = {
    AircraftAdsbCategory.A3,
    AircraftAdsbCategory.A4,
    AircraftAdsbCategory.A5,
}


def is_valid_aircraft_category() -> pl.Expr:
    return pl.col("category").is_in(
        [category.value for category in AircraftAdsbCategory]
    )


def is_likely_commercial_category() -> pl.Expr:
    return pl.col("category").is_in(
        [category.value for category in LIKELY_COMMERCIAL_CATEGORIES]
    )
