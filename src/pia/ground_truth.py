from data_engineering.ground_control import load_ground_control_mappings
from pia.identify_ladd_planes.pia_broadcast_adsb import get_pia_to_reg_broadcast_icaos


import polars as pl

def get_ground_truth_mappings():
    df_0 = load_ground_control_mappings().drop(["date"])
    df_1 = get_pia_to_reg_broadcast_icaos()
    df = pl.concat([df_0, df_1], how="vertical").unique(subset=["pia_icao"], keep="first")
    return df

def construct_ground_truth_mappings():
    df_0 = load_ground_control_mappings().drop(["date"])
    df_1 = get_pia_to_reg_broadcast_icaos()
    icaos_0 = set(df_0["pia_icao"])
    icaos_1 = set(df_1["pia_icao"])
    only_ground_control = icaos_0 - icaos_1
    only_broadcast = icaos_1 - icaos_0
    both = icaos_0 & icaos_1
    print(f"Only in ground_control mappings: {len(only_ground_control)}")
    print(f"Only in broadcast mappings: {len(only_broadcast)}")
    print(f"In both: {len(both)}")
    df = pl.concat([df_0, df_1], how="vertical").unique(subset=["pia_icao"], keep="first")
    return df, icaos_0, icaos_1, both