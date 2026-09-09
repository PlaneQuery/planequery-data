from datetime import date, datetime

import polars as pl

import data_engineering.eurocontrol.read as eurocontrol_read
from data_engineering.openairframes.read import get_openairframes_snapshot

def test_eurocontrol_processing():
    dt = date(2024, 6, 1)
    df_0 = pl.read_csv("/Volumes/T2-SSD/planequery/data/raw/eurocontrol/202406/extract/Flights_20240601_20240630.csv")
    number_of_regs_0 = len(df_0.get_column("AC Registration").unique())
    df_1 = eurocontrol_read.get_eurocontrol_flights(dt)
    number_of_regs_1 = len(df_1.get_column("registration").unique())
    assert 7923 / 12732 == number_of_regs_1 / number_of_regs_0