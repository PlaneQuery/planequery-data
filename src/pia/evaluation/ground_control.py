

from pathlib import Path

import polars as pl
from data_engineering.flights.flight_type import get_flights
from data_engineering.openairframes.read import scan_latest_faa_registration_icao_for_join
from datetime import date

from data_engineering.utils import OUTPUT_DIR

MAPPINGS_PATH = OUTPUT_DIR / "data/raw/groundcontrol/r-piaicao jacksweeney mappings.csv"
MAPPINGS_CACHE_PATH = OUTPUT_DIR / "data/intermediate/groundcontrol" / "r-piaicao jacksweeney mappings.parquet"
def get_ground_control_mappings():
    if MAPPINGS_CACHE_PATH.exists():
        df_mappings = pl.read_parquet(MAPPINGS_CACHE_PATH)
        return df_mappings
    df_mappings = pl.read_csv(MAPPINGS_PATH).rename(
        {
            "PIA ICAO": "pia_icao",
            "Tail Number": "tailnumber",
            "Date": "date",
        }
    ).with_columns(
        pl.col("pia_icao").str.strip_chars().str.to_lowercase(),
        pl.col("tailnumber").str.strip_chars().str.to_uppercase(),
        pl.col("date").str.strptime(pl.Date, strict=False),
    )
    df_real_icaos = scan_latest_faa_registration_icao_for_join().collect()

    df_mappings = df_mappings.join(
        df_real_icaos,
        left_on="tailnumber",
        right_on="registration",
        how="left",
    )
    df_mappings = df_mappings.rename({"real_icao": "icao", "tailnumber": "registration"})
    df_mappings = df_mappings.filter(pl.col("icao").is_not_null())
    df_flights = get_flights(date(2024, 1, 1), end_date = date(2026, 6, 8))
    df_mappings = df_mappings.filter(pl.col("icao").is_in(df_flights.get_column("icao").unique().to_list()))
    df_mappings.write_parquet(MAPPINGS_CACHE_PATH)    
    return df_mappings

if __name__ == "__main__":
    df_mappings = get_ground_control_mappings()
    print(df_mappings)