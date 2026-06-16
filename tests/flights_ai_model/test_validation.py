from datetime import datetime

import polars as pl

from flights_ai_model.validation import flights_df_comparison


def test_flights_df_comparison_matches_k_prefixed_us_airports_to_iata_codes():
    df1 = pl.DataFrame(
        [
            {
                "icao": "abc123",
                "takeoff_time": datetime(2026, 1, 1, 12, 0),
                "landing_time": datetime(2026, 1, 1, 13, 0),
                "takeoff_airport_ident": "KTUS",
                "landing_airport_ident": "TUS",
            }
        ]
    )
    df2 = pl.DataFrame(
        [
            {
                "icao": "abc123",
                "takeoff_time": datetime(2026, 1, 1, 12, 0),
                "landing_time": datetime(2026, 1, 1, 13, 0),
                "takeoff_airport_ident": "TUS",
                "landing_airport_ident": "KTUS",
            }
        ]
    )

    _df, stats = flights_df_comparison(df1, df2)

    assert stats["airport_ident_pct_match"] == 1.0
