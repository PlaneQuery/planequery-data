import polars as pl


def is_american_icao(col: str = "icao") -> pl.Expr:
    icao = pl.col(col).str.to_lowercase()
    return (icao >= "a00000") & (icao <= "afffff")


def pia_or_american_ladd() -> pl.Expr:
    return (
        pl.col("pia").fill_null(False)
        | (is_american_icao() & pl.col("ladd").fill_null(False))
    )


def pia_or_american_ladd_icao() -> pl.Expr:
    return pia_or_american_ladd().any().over("icao")
