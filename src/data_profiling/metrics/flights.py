from dataclasses import dataclass
from datetime import date
from typing import Literal, TypeAlias
import polars as pl

from aircraft.adsb_category import (
    AircraftAdsbCategory,
    is_valid_aircraft_category,
)
from data_engineering.flights.flight_type import get_flights


AircraftCategoryKey: TypeAlias = AircraftAdsbCategory | Literal["unknown"]
AircraftCategoryCounts: TypeAlias = dict[AircraftCategoryKey, int]

@dataclass
class CategoryDistribution:
    pass

@dataclass
class AirportDistribution:
    airport_types: str
    has_icao: bool

    # will have properties that get percentage 
    pass

class FlightsMetrics():
    def __init__(self, df: pl.DataFrame, day: date):
        self.day = day
        self.category_counts: AircraftCategoryCounts = {}
        self.category_counts["unknown"] = len(
            df.filter(~is_valid_aircraft_category().fill_null(False))
        )
        for category in AircraftAdsbCategory:
            self.category_counts[category] = len(
                df.filter(pl.col("category") == category.value)
            )

    def category_distribution_text(self) -> str:
        total = sum(self.category_counts.values())
        lines = [
            f"Aircraft categories for {self.day:%Y-%m-%d} ({total:,} flights):",
            f"  {'Category':<8} {'Count':>8} {'Percent':>9}",
        ]
        for category, count in self.category_counts.items():
            percentage = count / total if total else 0.0
            lines.append(
                f"  {str(category):<8} {count:>8,} {percentage:>8.2%}"
            )
        return "\n".join(lines)

    def to_vis(self) -> str:
        return self.category_distribution_text()

    day: date
    number: int
    number_likely_commercial: int
    number_likely_non_commercial: int
    number_unknown: int


    percent_likely_commercial: float
    percent_likely_non_commercial: float
    percent_unknown: float
