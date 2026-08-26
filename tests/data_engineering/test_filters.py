import polars as pl

from data_engineering.filters import (
    pia_or_american_ladd,
    pia_or_american_ladd_icao,
)


def _aircraft_rows() -> pl.DataFrame:
    return pl.DataFrame({
        "icao": ["a00001", "b00001", "c00001", "c00001"],
        "pia": [False, False, True, False],
        "ladd": [True, True, False, False],
    })


def test_pia_or_american_ladd_filters_individual_rows() -> None:
    result = _aircraft_rows().filter(pia_or_american_ladd())

    assert result["icao"].to_list() == ["a00001", "c00001"]


def test_pia_or_american_ladd_icao_filters_entire_icao_group() -> None:
    result = _aircraft_rows().filter(pia_or_american_ladd_icao())

    assert result["icao"].to_list() == ["a00001", "c00001", "c00001"]
