import polars as pl


def read_adsbx_readsb():
    paths = [
        "/Volumes/T2-SSD/planequery/data/processed/adsb-exchange/readsb-hist/parquet/year=2026/month=03/day=01/part-00000.parquet",
        "/Volumes/T2-SSD/planequery/data/processed/adsb-exchange/readsb-hist/parquet/year=2026/month=03/day=01/part-00001.parquet",
    ]

    columns = [
        "time",
        "icao",
        "lat",
        "lon",
    ]

    df = pl.read_parquet(
        paths,
        columns=columns,
    )
    return df