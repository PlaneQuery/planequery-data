import polars as pl

CALLSIGN_PATTERN = r"^[A-Z0-9]{1,8}$"   
def is_valid_callsign(column: str = "callsign") -> pl.Expr:
    return pl.col(column).str.strip_chars().str.contains(CALLSIGN_PATTERN)