from datetime import date
from data_engineering.ground_control import load_ground_control_mappings
from data_engineering.openairframes.read import get_openairframes_faa_df
from data_engineering.utils import OUTPUT_DIR
from pia.identify_ladd_planes.pia_broadcast_adsb import get_pia_to_reg_broadcast_icaos
import polars as pl
import os
from pia.identify_ladd_planes.pia_callsign_switch import get_pia_callsign_switch_icaos

OUTPUT_PATH = OUTPUT_DIR / "cache/pia"

def get_ground_truth_mappings(use_cache = True, start_date = date(2024,1,1)) -> pl.DataFrame:
    start_date_str = start_date.isoformat()
    output_path = OUTPUT_PATH / f"ground_truth_mappings_start_date={start_date_str}.parquet"
    if use_cache and output_path.exists():
        return pl.read_parquet(output_path)
    df_0 = load_ground_control_mappings().filter(pl.col("date") > start_date).drop(["date"])
    df_1 = get_pia_to_reg_broadcast_icaos()
    df_2 = get_pia_callsign_switch_icaos()
    df = pl.concat([df_0, df_1, df_2], how="vertical").unique(subset=["pia_icao"], keep="first")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    df.write_parquet(output_path)
    return df


BASE_PATH = OUTPUT_DIR / "data/raw/adsblol/parquet_output/v6"
def get_unique_pia_icaos_from_adsblol() -> pl.Series:
    pia_icaos = (
        pl.scan_parquet(
            str(BASE_PATH / "year=*/month=*/day=*/icao_bucket=5/data.parquet"),
            hive_partitioning=True,
            missing_columns="insert",
        )
        .filter(pl.col("pia"))
        .select(pl.col("icao").str.to_lowercase())
        .unique()
        .sort("icao")
        .collect()
        .get_column("icao")
    )
    assert_pia_icaos_not_in_faa(pia_icaos)
    return pia_icaos


def assert_pia_icaos_not_in_faa(pia_icaos: pl.Series) -> None:
    faa_icaos = set(
        get_openairframes_faa_df()
        .get_column("transponder_code_hex")
        .drop_nulls()
        .str.strip_chars()
        .str.to_lowercase()
    )
    overlapping_icaos = sorted(set(pia_icaos.str.to_lowercase()) & faa_icaos)
    assert not overlapping_icaos, (
        "PIA ICAOs found in FAA transponder_code_hex: "
        f"{overlapping_icaos[:20]}"
    )


def create_pia_csv() -> pl.DataFrame:
    output_path = OUTPUT_DIR / "data" /"pia" / "all_pia_icaos.csv"
    pia_icaos = get_unique_pia_icaos_from_adsblol()
    assert_pia_icaos_not_in_faa(pia_icaos)
    df = pl.DataFrame({"pia_icao": pia_icaos})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(output_path)
    df.write_csv(output_path)
    return df

if __name__ == "__main__":
    print(create_pia_csv())