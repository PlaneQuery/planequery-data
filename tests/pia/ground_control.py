from pathlib import Path
import polars as pl
from datetime import date

from data_engineering.openairframes.read import scan_latest_faa_registration_icao_for_join

def load_ground_control_mappings():
    mappings_path = Path("/Users/jonahgoode/Documents/PlaneQuery/r-piaicao jacksweeney mappings.csv")

    df_mappings = pl.read_csv(mappings_path).rename(
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

    df_mappings = df_mappings.filter(pl.col("real_icao").is_not_null())
    df_mappings = df_mappings.rename({"tailnumber": "registration"})
    return df_mappings

if __name__ == "__main__":
    df_mappings = load_ground_control_mappings()
    print(df_mappings)