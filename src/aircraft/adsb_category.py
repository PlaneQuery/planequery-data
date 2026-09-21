import polars as pl


AIRCRAFT_ADSB_CATEGORIES = [
    "A1",  # Light aircraft
    "A2",  # Small aircraft
    "A3",  # Large aircraft
    "A4",  # High-vortex large aircraft
    "A5",  # Heavy aircraft
    "A6",  # High-performance aircraft
    "A7",  # Rotorcraft
    "B1",  # Glider / sailplane
    "B2",  # Lighter-than-air
    "B3",  # Parachutist / skydiver
    "B4",  # Ultralight / hang-glider / paraglider
    "B6",  # Unmanned aerial vehicle
    "B7",  # Space / trans-atmospheric vehicle
]

LIKELY_COMMERCIAL_CATEGORIES = {
    "A3",
    "A4",
    "A5",
}


def is_valid_aircraft_category() -> pl.Expr:
    return pl.col("category").is_in(AIRCRAFT_ADSB_CATEGORIES)


def is_commercial_category() -> pl.Expr:
    return pl.col("category").is_in(LIKELY_COMMERCIAL_CATEGORIES)
