import argparse
import os
from datetime import date
from pathlib import Path

os.environ.setdefault("OUTPUT_DIR", str(Path.cwd()))

import polars as pl

from data_engineering.adsbx.get_flights import get_adsbx_flights_for_day
from data_engineering.flights.flight_type import get_flights
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from flights_ai_model.validation import flights_df_comparison

DEFAULT_TARGET_DATE = date(2026, 3, 1)
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])


def read_opensky_flights(target_date: date) -> pl.DataFrame:
    return pl.read_parquet(f"/Volumes/T2-SSD/planequery/data/flights/opensky/v1/{target_date:%Y-%m-%d}.parquet")


def default_stats_path(target_date: date) -> Path:
    return OUTPUT_DIR / "data" / "experiments" / "comparison" / f"{target_date:%Y-%m-%d}_stats.csv"


def main() -> None:
    target_date = date(2026,3,1)
    stats_output = default_stats_path(target_date)

    df_sfdps_flights = get_sfdps_flights_day(target_date)

    df_model_flights = get_flights(target_date)
    df_opensky_flights = read_opensky_flights(target_date)
    df_adsbx_flights = get_adsbx_flights_for_day(target_date)

    rows = []
    for df, name in [
        (df_model_flights, "Model Flights"),
        (df_opensky_flights, "OpenSky Flights"),
        (df_adsbx_flights, "ADSBX Flights"),
    ]:
        _, stats = flights_df_comparison(df, df_sfdps_flights)
        print(f"Stats for {name}: {stats}")
        rows.append({"date": target_date.isoformat(), "source": name, **stats})

    stats_output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(stats_output)
    print(f"Stats CSV saved to {stats_output}")


if __name__ == "__main__":
    main()
