from dataclasses import dataclass, field
from datetime import date
from typing import Literal, TypeAlias
import polars as pl

from aircraft.adsb_category import (
    AircraftAdsbCategory,
    is_valid_aircraft_category,
)
from airports.polars_exprs import (
    airport_has_iata_code_expr,
    airport_has_icao_code_expr,
    airport_size_expr,
)
from airports.runtime import airport_catalog_for_date


AircraftCategoryKey: TypeAlias = AircraftAdsbCategory | Literal["unknown"]
AircraftCategoryCounts: TypeAlias = dict[AircraftCategoryKey, int]

@dataclass
class CategoryDistribution:
    pass

@dataclass
class AirportDistribution:
    total: int
    size_counts: dict[str, int]
    has_icao_code_count: int
    has_iata_code_count: int


@dataclass
class FlightCohort:
    name: str
    df: pl.DataFrame
    children: list["FlightCohort"] = field(default_factory=list)

    @property
    def count(self) -> int:
        return self.df.height

    def add_child(self, name: str, df: pl.DataFrame) -> "FlightCohort":
        child = FlightCohort(name=name, df=df)
        self.children.append(child)
        return child

    def add_subset(self, name: str, predicate: pl.Expr) -> "FlightCohort":
        return self.add_child(name, self.df.filter(predicate))

    def to_text(
        self,
        *,
        root_count: int | None = None,
        parent_count: int | None = None,
        depth: int = 0,
    ) -> str:
        if root_count is None:
            root_count = self.count

        line = f"{'  ' * depth}{self.name}: {self.count:,}"
        if parent_count is not None:
            parent_percentage = self.count / parent_count if parent_count else 0.0
            total_percentage = self.count / root_count if root_count else 0.0
            if parent_count == root_count:
                line += f" ({total_percentage:.2%} of total)"
            else:
                line += (
                    f" ({parent_percentage:.2%} of parent, "
                    f"{total_percentage:.2%} of total)"
                )

        lines = [line]
        for child in self.children:
            lines.append(
                child.to_text(
                    root_count=root_count,
                    parent_count=self.count,
                    depth=depth + 1,
                )
            )
        return "\n".join(lines)

class FlightsMetrics():
    def __init__(self, df: pl.DataFrame, day: date):
        self.day = day
        self.generate_category_statistics(df)
        self.generate_airport_statistics(df)

    def generate_airport_statistics(self, df: pl.DataFrame) -> None:
        airport_catalog = airport_catalog_for_date(self.day)
        df = df.with_columns(
            expr
            for prefix in ("takeoff", "landing")
            for expr in (
                airport_size_expr(airport_catalog, f"{prefix}_airport_id"),
                airport_has_icao_code_expr(
                    airport_catalog,
                    f"{prefix}_airport_id",
                ),
                airport_has_iata_code_expr(
                    airport_catalog,
                    f"{prefix}_airport_id",
                ),
            )
        )
        self.airport_statistics: dict[str, AirportDistribution] = {}
        for prefix in ("takeoff", "landing"):
            size_counts = dict(
                df.get_column(f"{prefix}_airport_size")
                .fill_null("unknown")
                .value_counts()
                .iter_rows()
            )
            self.airport_statistics[prefix] = AirportDistribution(
                total=df.height,
                size_counts=size_counts,
                has_icao_code_count=int(
                    df.get_column(f"{prefix}_airport_has_icao_code").sum() or 0
                ),
                has_iata_code_count=int(
                    df.get_column(f"{prefix}_airport_has_iata_code").sum() or 0
                ),
            )

    def generate_category_statistics(self, df: pl.DataFrame) -> None:
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

    def airport_statistics_text(self) -> str:
        lines = [
            f"Airport statistics for {self.day:%Y-%m-%d}:",
        ]
        size_order = (
            "large_airport",
            "medium_airport",
            "small_airport",
            "unknown",
        )
        for prefix, statistics in self.airport_statistics.items():
            lines.extend([
                f"  {prefix.title()} airports ({statistics.total:,} flight endpoints):",
                f"    {'Attribute':<20} {'Count':>8} {'Percent':>9}",
            ])
            for airport_size in size_order:
                count = statistics.size_counts.get(airport_size, 0)
                percentage = count / statistics.total if statistics.total else 0.0
                lines.append(
                    f"    {airport_size:<20} {count:>8,} {percentage:>8.2%}"
                )
            for label, count in (
                ("has_icao_code", statistics.has_icao_code_count),
                ("has_iata_code", statistics.has_iata_code_count),
            ):
                percentage = count / statistics.total if statistics.total else 0.0
                lines.append(f"    {label:<20} {count:>8,} {percentage:>8.2%}")
        return "\n".join(lines)

    def to_vis(self) -> str:
        return "\n\n".join([
            self.category_distribution_text(),
            self.airport_statistics_text(),
        ])

    day: date
    number: int
    number_likely_commercial: int
    number_likely_non_commercial: int
    number_unknown: int


    percent_likely_commercial: float
    percent_likely_non_commercial: float
    percent_unknown: float
