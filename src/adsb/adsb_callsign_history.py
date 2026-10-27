import polars as pl
from aircraft.callsign import is_valid_callsign
import polars as pl

def get_callsign_history(df):
    assert df.select(["icao", "time"]).equals(
        df.sort(["icao", "time"]).select(["icao", "time"])
    )

    df_history = (
        df
        .with_columns(
            pl.when(is_valid_callsign().fill_null(False))
            .then(pl.col("callsign"))
            .otherwise(None)
            .alias("callsign")
        )
        .with_columns(
            pl.col("callsign")
            .forward_fill()
            .backward_fill()
            .over("icao")
        )
        # Identify a new callsign interval within each ICAO.
        .with_columns(
            (
                pl.col("callsign")
                != pl.col("callsign").shift().over("icao")
            )
            .fill_null(True)
            .alias("new_range")
        )
        .with_columns(
            pl.col("new_range")
            .cum_sum()
            .over("icao")
            .alias("range_id")
        )
        .group_by(["icao", "range_id"], maintain_order=True)
        .agg(
            pl.col("time").first().alias("start_time"),
            pl.col("time").last().alias("last_seen_time"),
            pl.col("callsign").first(),
        )
        .drop("range_id")
    )

    return df_history

'''
TODO: make tests for these 2 icaos on 2026-03-01
a008c5 should succeed
a19a21 should fail
'''