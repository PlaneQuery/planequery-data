from pathlib import Path

from data_engineering.opensky.read_opensky_trino_states import (
    align_to_adsb_parquet_schema,
    join_lf_with_openairframes,
    scan_normalized_opensky_state_vectors_for_day,
)
from data_engineering.openairframes.read import scan_latest_icao_info_for_join


CURRENT_DAY = "2026-03-01"
OUTPUT_PATH = Path(
    "/Volumes/T2-SSD/planequery/data/raw/opensky/state_vectors_parquet_output/v5/"
    "year=2026/month=03/day=01/part-00000.parquet"
)


if __name__ == "__main__":
    df = scan_normalized_opensky_state_vectors_for_day(CURRENT_DAY)
    df_openairframes = scan_latest_icao_info_for_join()
    df = join_lf_with_openairframes(df, df_openairframes)
    df = align_to_adsb_parquet_schema(df)
    df.sink_parquet(
        OUTPUT_PATH,
        compression="zstd",
        maintain_order=False,
        mkdir=True,
    )
