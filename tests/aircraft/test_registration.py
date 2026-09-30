import polars as pl
import pytest

from aircraft.registration import (
    is_valid_n_number,
    is_valid_n_number_expr,
    is_valid_registration_callsign,
    is_valid_registration_callsign_expr,
)


@pytest.mark.parametrize(
    "value",
    ["N1", "N12345", "N1A", "N1234Z", "N1AA", "N123AZ", " n123ab "],
)
def test_valid_n_numbers(value):
    assert is_valid_n_number(value)


@pytest.mark.parametrize(
    "value",
    [None, "", "N0", "N0123", "N123456", "N1234AB", "N123I", "N123O", "NKS108"],
)
def test_invalid_n_numbers(value):
    assert not is_valid_n_number(value)


def test_n_number_polars_expression_normalizes_values():
    result = pl.DataFrame(
        {"registration": ["n123ab", " N123 ", "N1234AB", None]}
    ).select(is_valid_n_number_expr("registration").alias("valid"))

    assert result.get_column("valid").to_list() == [True, True, False, False]


@pytest.mark.parametrize("value", ["N123AB", "C-GABC", "G-ABCD", "ec-abc"])
def test_valid_registration_callsigns(value):
    assert is_valid_registration_callsign(value)


@pytest.mark.parametrize("value", [None, "NKS108", "N123A4", "CGABC", "-ABC"])
def test_invalid_registration_callsigns(value):
    assert not is_valid_registration_callsign(value)


def test_registration_callsign_polars_expression():
    result = pl.DataFrame(
        {"callsign": ["N123AB", " c-gabc ", "NKS108", None]}
    ).select(is_valid_registration_callsign_expr("callsign").alias("valid"))

    assert result.get_column("valid").to_list() == [True, True, False, False]
