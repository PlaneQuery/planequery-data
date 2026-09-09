from datetime import date, datetime

import polars as pl

import data_engineering.eurocontrol.read as eurocontrol_read

def test_eurocontrol_processing():
    dt = date(2024, 6, 1)
    df_0 = pl.read_csv("/Volumes/T2-SSD/planequery/data/raw/eurocontrol/202406/extract/Flights_20240601_20240630.csv")
    df_0 = df_0.with_columns(eurocontrol_read.datetime_expr("ACTUAL OFF BLOCK TIME", "takeoff_time"))
    df_0 = df_0.filter(pl.col("takeoff_time").dt.date() == dt)
    flights_len_0 = len(df_0)
    number_of_regs_0 = len(df_0.get_column("AC Registration").unique())
    df_1 = eurocontrol_read.get_eurocontrol_flights(dt)
    number_of_regs_1 = len(df_1.get_column("registration").unique())
    flights_len_1 = len(df_1)
    print(f"Flights len 0: {flights_len_0}, Flights len 1: {flights_len_1}")
    assert 7923 / 12732 == number_of_regs_1 / number_of_regs_0