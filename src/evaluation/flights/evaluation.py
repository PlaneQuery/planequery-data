import argparse
from datetime import UTC, date, datetime
from pathlib import Path
import polars as pl
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.bts.read_bts_ontime_to_flights import get_bts_flights_for_day
from data_engineering.flights.flight_type import get_flights
from flights.flights_match_adsb import get_matching_icaos_in_flights
from utils import current_commit_hash
from data_engineering.adsbx.get_flights import get_adsbx_flights_for_day
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from flights_ai_model.validation import flights_df_comparison
from data_engineering.utils import OUTPUT_DIR

DEFAULT_TARGET_DATE = date(2026, 3, 1)

def read_opensky_flights(target_date: date) -> pl.DataFrame:
    return pl.read_parquet(f"/Volumes/T2-SSD/planequery/data/flights/opensky/v1/{target_date:%Y-%m-%d}.parquet")

def default_stats_path() -> Path:
    run_time = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M")
    commit_hash = current_commit_hash()
    filename = f"{run_time}_{commit_hash}_flights_evaluation.csv"
    return OUTPUT_DIR / "data" / "experiments" / "comparison" / filename

def source_flights_reader(src: str):
    if src == "adsbx":
        return get_adsbx_flights_for_day
    if src == "model":
        return get_flights
    if src == "sfdps":
        return get_sfdps_flights_day
    if src == "bts":
        return get_bts_flights_for_day
    if src =="opensky":
        return read_opensky_flights
    raise ValueError(f"Unknown flight source: {src}")

def main(test_src: str | list[str], gold_src: str | list[str], test_dates: date | list[date], use_all_icaos = False):
    if isinstance(test_src, str):
        test_src = [test_src]
    if isinstance(gold_src, str):
        gold_src = [gold_src]
    if isinstance(test_dates, date):
        test_dates = [test_dates]

    rows = []
    for src in test_src:
        get_source_flights = source_flights_reader(src)
        for gold in gold_src:
            get_gold_flights = source_flights_reader(gold)
            for test_date in test_dates:
                df_source = get_source_flights(test_date)
                df_gold = get_gold_flights(test_date)
                if not use_all_icaos:
                    df_adsb = read_adsb(test_date)
                    df_gold = get_matching_icaos_in_flights(df_gold, df_adsb)
                    df_source = df_source.filter(pl.col("icao").is_in(df_gold.get_column("icao").implode()))
                _, stats = flights_df_comparison(df_source, df_gold)
                rows.append({"test_date": test_date.isoformat(), "test_src": src, "gold_src": gold, "use_all_icaos": use_all_icaos, **stats})
                print(f"Stats for {src}:{gold}: {stats}")

    stats_output = default_stats_path()
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(stats_output)
    print(f"Stats CSV saved to {stats_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-src", nargs="+", default=["adsbx"])
    parser.add_argument("--gold-src", nargs="+", default=["bts"])
    parser.add_argument("--date", nargs="+", default=[DEFAULT_TARGET_DATE.isoformat()])
    parser.add_argument("--use_all_icaos", action="store_true", default=False)
    args = parser.parse_args()

    main(
        test_src=args.test_src,
        gold_src=args.gold_src,
        test_dates=[date.fromisoformat(d) for d in args.date],
        use_all_icaos=args.use_all_icaos,
    )
