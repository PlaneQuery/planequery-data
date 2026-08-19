from data_engineering.ground_control import load_ground_control_mappings
from data_engineering.utils import OUTPUT_DIR
from pia.identify_ladd_planes.pia_broadcast_adsb import get_pia_to_reg_broadcast_icaos
import polars as pl
import os
from pia.identify_ladd_planes.pia_callsign_switch import get_pia_callsign_switch_icaos

OUTPUT_PATH = OUTPUT_DIR / "cache/pia"

def get_ground_truth_mappings(use_cache = True):
    output_path = OUTPUT_PATH / "ground_truth_mappings.parquet"
    if use_cache and output_path.exists():
        return pl.read_parquet(output_path)
    df_0 = load_ground_control_mappings().drop(["date"])
    df_1 = get_pia_to_reg_broadcast_icaos()
    df_2 = get_pia_callsign_switch_icaos()
    df = pl.concat([df_0, df_1, df_2], how="vertical").unique(subset=["pia_icao"], keep="first")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    df.write_parquet(output_path)
    return df

def construct_ground_truth_mappings_with_stats():
    df_0 = load_ground_control_mappings().drop(["date"])
    df_1 = get_pia_to_reg_broadcast_icaos()
    df_2 = get_pia_callsign_switch_icaos()
    icaos_0 = set(df_0["pia_icao"])
    icaos_1 = set(df_1["pia_icao"])
    icaos_2 = set(df_2["pia_icao"])
    only_ground_control = icaos_0 - icaos_1
    only_broadcast = icaos_1 - icaos_0
    only_callsign_switch = icaos_2 - icaos_0 - icaos_1
    both = icaos_0 & icaos_1
    print(f"Only in ground_control mappings: {len(only_ground_control)}")
    print(f"Only in broadcast mappings: {len(only_broadcast)}")
    print(f"Only in pia->callsign_real mappings: {len(only_callsign_switch)}")
    print(f"In both: {len(both)}")
    df = pl.concat([df_0, df_1, df_2], how="vertical").unique(subset=["pia_icao"], keep="first")
    return df, icaos_0, icaos_1, icaos_2, both
    # TODO: Add assert that a pia_icao is only ever mapped to 1 real icao. 

if __name__ == "__main__":
    df = get_ground_truth_mappings(use_cache=False)
    print(df)