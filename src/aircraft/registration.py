import re

import polars as pl


_N_NUMBER_BODY = (
    r"N(?:[1-9][0-9]{0,4}|"
    r"[1-9][0-9]{0,3}[A-HJ-NP-Z]|"
    r"[1-9][0-9]{0,2}[A-HJ-NP-Z]{2})"
)
N_NUMBER_PATTERN = rf"^{_N_NUMBER_BODY}$"

_HYPHENATED_REGISTRATION_BODY = r"[A-Z]{1,3}-[A-Z0-9]{2,5}"
REGISTRATION_CALLSIGN_PATTERN = (
    rf"^(?:{_HYPHENATED_REGISTRATION_BODY}|{_N_NUMBER_BODY})$"
)

_N_NUMBER_RE = re.compile(N_NUMBER_PATTERN)
_REGISTRATION_CALLSIGN_RE = re.compile(REGISTRATION_CALLSIGN_PATTERN)


def is_valid_n_number(value: str | None) -> bool:
    """Return whether a value is a valid, normalized FAA N-number."""
    if value is None:
        return False
    return _N_NUMBER_RE.fullmatch(value.strip().upper()) is not None


def is_valid_n_number_expr(column: str) -> pl.Expr:
    """Return a Polars expression validating FAA N-numbers in ``column``."""
    return (
        pl.col(column)
        .str.strip_chars()
        .str.to_uppercase()
        .str.contains(N_NUMBER_PATTERN)
        .fill_null(False)
    )


def is_valid_registration_callsign(value: str | None) -> bool:
    """Validate an N-number or a hyphenated international registration."""
    if value is None:
        return False
    return _REGISTRATION_CALLSIGN_RE.fullmatch(value.strip().upper()) is not None


def is_valid_registration_callsign_expr(column: str) -> pl.Expr:
    """Return a Polars expression validating registration-like callsigns."""
    return (
        pl.col(column)
        .str.strip_chars()
        .str.to_uppercase()
        .str.contains(REGISTRATION_CALLSIGN_PATTERN)
        .fill_null(False)
    )
