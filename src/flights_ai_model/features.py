import os
from datetime import datetime, date
from pathlib import Path
import numpy as np
import polars as pl
import glob
import json
import tempfile
import boto3
from datetime import timedelta
from data_engineering.adsb.read_adsb import get_icaos_in_adsb, read_adsb, exclude_invalid_icaos, read_icaos_from_adsb
from data_engineering.bts.read_bts_ontime_to_flights import create_flights_from_bts
from data_engineering.flights.flights_and_adsb_match import get_icaos_with_perfect_match
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from data_engineering.utils import is_ecs, s3_key_exists
from data_engineering.utils import OUTPUT_DIR, S3_BUCKET
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
INFERENCE_FEATURES_VERSION = "v2"
TRAINING_FEATURES_VERSION = "v17"
FEATURES_DIR = OUTPUT_DIR / "data/features"
TRAINING_FEATURES_DIR = FEATURES_DIR / "training" / TRAINING_FEATURES_VERSION
INFERENCE_FEATURES_DIR = FEATURES_DIR / "inference" / INFERENCE_FEATURES_VERSION
os.makedirs(TRAINING_FEATURES_DIR, exist_ok=True)
os.makedirs(INFERENCE_FEATURES_DIR, exist_ok=True)

NULLABLE_COLS = ["baro_altitude_ft", "ground_speed_kt", "track_deg", "above_ground_level_closest_airport"]

ZSCORE_COLS  = ["baro_altitude_ft", "ground_speed_kt", "track_deg", "time_delta_s", "above_ground_level_closest_airport"]

ZSCORE_STATS_S3_KEY = f"features/training/{TRAINING_FEATURES_VERSION}/zscore_stats.json"

s3 = boto3.client("s3")

def _get_zscore_stats() -> dict:
    path = TRAINING_FEATURES_DIR / "zscore_stats.json"
    os.makedirs(path.parent, exist_ok=True)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Try downloading from S3 before recomputing.
    s3_key = str(path.relative_to(OUTPUT_DIR))
    if s3_key_exists(s3, S3_BUCKET, s3_key):
        s3.download_file(S3_BUCKET, s3_key, str(path))
        with open(path) as f:
            return json.load(f)
    parquet_paths = glob.glob(str(TRAINING_FEATURES_DIR / 'year=*/month=*/day=*/part-0.parquet'), recursive=False)
    print(f"Found {len(parquet_paths)} parquet files")
    col_values: dict[str, list[np.ndarray]] = {c: [] for c in ZSCORE_COLS}
    for path in parquet_paths:
        df = pl.read_parquet(path, columns=ZSCORE_COLS)
        for c in ZSCORE_COLS:
            col_values[c].append(df.get_column(c).to_numpy())
    stats = {}
    for c in ZSCORE_COLS:
        all_vals = np.concatenate(col_values[c])
        stats[c] = {"mean": float(all_vals.mean()), "std": float(all_vals.std())}
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats

def apply_zscore(df: pl.DataFrame) -> pl.DataFrame:
    ZSCORE_STATS = _get_zscore_stats()
    return df.with_columns([
        ((pl.col(c) - ZSCORE_STATS[c]["mean"]) / ZSCORE_STATS[c]["std"]).alias(c)
        for c in ZSCORE_STATS
    ])

def _training_features_path(current_date: date) -> Path:
    return (
        TRAINING_FEATURES_DIR
        / f"year={current_date:%Y}"
        / f"month={current_date:%m}"
        / f"day={current_date:%d}"
        / "part-0.parquet"
    )
def inference_features_path(current_date: date) -> Path:
    return (
        INFERENCE_FEATURES_DIR
        / f"year={current_date:%Y}"
        / f"month={current_date:%m}"
        / f"day={current_date:%d}"
        / "part-0.parquet"
    )
CUTOFF_TIME = pl.duration(minutes=30)
def join_flights_and_adsb(df_adsb: pl.DataFrame, df_flights: pl.DataFrame) -> pl.DataFrame:
    icaos = df_flights.get_column("icao").unique()

    def label_icao(icao: str) -> pl.DataFrame:
        df = df_adsb.filter(pl.col("icao") == icao)
        flights = df_flights.filter(pl.col("icao") == icao)

        # Build a label series by joining on time ranges
        df = df.with_columns([
            pl.lit(0).alias("label"),
        ])

        for row in flights.rows(named=True):
            in_flight = (pl.col("time") >= row["takeoff_time"]) & (pl.col("time") <= row["landing_time"])
            # Find closest time >= takeoff_time
            takeoff_candidates = df.filter(pl.col("time") >= row["takeoff_time"]).get_column("time")
            landing_candidates = df.filter(pl.col("time") <= row["landing_time"]).get_column("time")

            if takeoff_candidates.is_empty() or landing_candidates.is_empty():
                continue

            actual_takeoff = takeoff_candidates.min()
            actual_landing = landing_candidates.max()

            in_flight = (pl.col("time") >= actual_takeoff) & (pl.col("time") <= actual_landing)

            df = df.with_columns([
                pl.when(in_flight & (pl.col("time") == actual_takeoff)).then(1)
                  .when(in_flight & (pl.col("time") == actual_landing)).then(3)
                  .when(in_flight).then(2)
                  .otherwise(pl.col("label"))
                  .alias("label"),
            ])

        # remove rows in df before first takeoff and after last landing
        if not flights.is_empty():
            first_takeoff = flights.get_column("takeoff_time").min()
            last_landing = flights.get_column("landing_time").max()
            df = df.filter((pl.col("time") >= (first_takeoff - CUTOFF_TIME)) & (pl.col("time") <= (last_landing + CUTOFF_TIME)))
        return df

    dfs = []
    for icao in icaos:
        df_labeled = label_icao(icao)
        dfs.append(df_labeled)
    df = pl.concat(dfs).sort(["icao", "time"])
    return df

def create_training_features_for_day(current_date: date) -> pl.DataFrame:
    cache_path = _training_features_path(current_date)
    if cache_path.exists():
        return pl.read_parquet(cache_path)
    df_adsb = read_adsb(current_date)
    df_flights = get_sfdps_flights_day(current_date)
    df_flights = df_flights.filter(
        (pl.col("takeoff_time").dt.date() == current_date) &
        (pl.col("landing_time").dt.date() == current_date)
    )
    icaos = get_icaos_with_perfect_match(df_flights, df_adsb)
    df_flights = df_flights.filter(pl.col("icao").is_in(icaos))
    df_features = join_flights_and_adsb(df_adsb, df_flights)
    df_features = _to_features(df_features)
    os.makedirs(cache_path.parent, exist_ok=True)
    df_features.write_parquet(cache_path)
    return df_features

BATCH_SIZE = 1000 # this uses 3GB of memory. 
def create_inference_features_for_day(target_date: date, write_to_s3 = False):
    parquet_path = INFERENCE_FEATURES_DIR / f"year={target_date.year}" / f"month={target_date.month:02d}" / f"day={target_date.day:02d}" / "part-0.parquet"
    os.makedirs(parquet_path.parent, exist_ok=True)
    dates = [target_date - timedelta(days=1), target_date, target_date + timedelta(days=1)]
    icaos = []
    for date in dates:
        icaos.extend(get_icaos_in_adsb(date))
    icaos = list(set(icaos))
    with tempfile.TemporaryDirectory() as tmpdir:
        for batch_num, i in enumerate(range(0, len(icaos), BATCH_SIZE)):
            icao_subset = icaos[i:i+BATCH_SIZE]
            dfs = []
            for date in dates:
                df = read_icaos_from_adsb(date, icao_subset)
                dfs.append(df)
            df_adsb = pl.concat(dfs)
            df_features = create_inference_features(df_adsb)
            df_features.write_parquet(Path(tmpdir) / f"part_{batch_num:05d}.parquet")
            print(f"Processed {min(i+BATCH_SIZE, len(icaos))}/{len(icaos)} ICAOs")
        pl.scan_parquet(str(Path(tmpdir) / "part_*.parquet")).sink_parquet(parquet_path)
    if write_to_s3:
        s3_key = str(parquet_path.relative_to(OUTPUT_DIR))
        s3.upload_file(str(parquet_path), S3_BUCKET, s3_key)

def create_inference_features(df_adsb: pl.DataFrame) -> pl.DataFrame:
    df_features = _to_features(df_adsb)
    df_features = apply_zscore(df_features)
    df_features = df_features.sort(["icao", "time"])
    return df_features
def create_inference_features_parquet_for_day(current_date: date):
    parquet_path = inference_features_path(current_date)
    os.makedirs(parquet_path.parent, exist_ok=True)
    if parquet_path.exists():
        return
    s3_key = str(parquet_path.relative_to(OUTPUT_DIR))
    if s3_key_exists(s3, S3_BUCKET, s3_key):
        print(f"Download {s3_key} from S3 bucket {S3_BUCKET}")
        s3.download_file(S3_BUCKET, s3_key, str(parquet_path))
        return pl.read_parquet(parquet_path)
    create_inference_features_for_day(current_date)

def read_icaos_from_features(current_date: date, icaos: list[str]) :
    create_inference_features_parquet_for_day(current_date)
    parquet_path = inference_features_path(current_date)
    df = pl.scan_parquet(parquet_path).filter(pl.col("icao").is_in(icaos)).collect()
    return df

def get_icaos_in_features(current_date: date) -> list[str]:
    create_inference_features_parquet_for_day(current_date)
    parquet_path = inference_features_path(current_date)
    df = pl.scan_parquet(parquet_path).select("icao").unique().collect()
    return df["icao"].sort().to_list()
def _to_features(df: pl.DataFrame) -> pl.DataFrame:
    print(f"Start compute elevation from closest airport")
    df = elevation_from_closest_airport(df)
    print(f"Finished compute elevation from closest airport")
    df = (
        df
        .sort(["icao", "time"])
        .with_columns(
            [pl.col(c).forward_fill().over("icao") for c in NULLABLE_COLS] +
            [pl.col("time").diff().over("icao").dt.total_seconds().fill_null(0).alias("time_delta_s")]
        )
        .with_columns(
            [pl.col(c).backward_fill().over("icao").fill_null(0.0) for c in NULLABLE_COLS]  # fill_null not normally hit
        )
    )
    df = callsign_feature(df)
    # lat_rad = np.radians(df.get_column("lat").to_numpy())
    # lon_rad = np.radians(df.get_column("lon").to_numpy())
    # df = df.with_columns(
    #     pl.Series("lat_sin", np.sin(lat_rad), dtype=pl.Float64),
    #     pl.Series("lat_cos", np.cos(lat_rad), dtype=pl.Float64),
    #     pl.Series("lon_sin", np.sin(lon_rad), dtype=pl.Float64),
    #     pl.Series("lon_cos", np.cos(lon_rad), dtype=pl.Float64),
    # )
    return df.with_columns(
        pl.col("on_ground").cast(pl.Int8),
        pl.col("time_delta_s").log1p().alias("time_delta_s"),
    )

def callsign_feature(df: pl.DataFrame) -> pl.DataFrame:
    '''
    callsign_changed_from_prev
    '''
    df = df.with_columns(
    pl.when(pl.col("callsign") == "")
      .then(None)
      .otherwise(pl.col("callsign"))
      .alias("callsign")
    )
    df = df.with_columns(pl.col("callsign").forward_fill().over("icao"))
    df = df.with_columns(
        (pl.col("callsign") != pl.col("callsign").shift(1).over("icao")).cast(pl.Int8).fill_null(0).alias("callsign_changed_from_prev")
    )
    return df

from airports.airportLookup import AirportLookup
airport_lookup = AirportLookup()
_airport_elevations = np.array([a.elevation_ft for a in airport_lookup._airports], dtype=np.float64)

def elevation_from_closest_airport(df: pl.DataFrame) -> pl.DataFrame:
    lat = df["lat"].to_numpy(allow_copy=True)
    lon = df["lon"].to_numpy(allow_copy=True)
    alt = df["baro_altitude_ft"].to_numpy(allow_copy=True)

    valid = ~(np.isnan(lat) | np.isnan(lon) | np.isnan(alt))
    agl_full = np.full(len(df), np.nan)

    if valid.any():
        lat_v, lon_v, alt_v = lat[valid], lon[valid], alt[valid]

        # Deduplicate KDTree queries by rounding to ~5km grid cells
        GRID = 0.05
        lat_r = np.round(lat_v / GRID) * GRID
        lon_r = np.round(lon_v / GRID) * GRID
        unique_cells, inverse = np.unique(
            np.column_stack([lat_r, lon_r]), axis=0, return_inverse=True
        )

        lat_u_rad = np.radians(unique_cells[:, 0])
        lon_u_rad = np.radians(unique_cells[:, 1])
        xyz_q = np.column_stack([
            np.cos(lat_u_rad) * np.cos(lon_u_rad),
            np.cos(lat_u_rad) * np.sin(lon_u_rad),
            np.sin(lat_u_rad),
        ])
        _, indices = airport_lookup._kd_tree.query(xyz_q, workers=-1)
        agl_full[valid] = np.maximum(0.0, alt_v - _airport_elevations[indices[inverse]])

    return df.with_columns(
        pl.Series("above_ground_level_closest_airport", agl_full, dtype=pl.Float64).fill_nan(None)
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=False, default="training", choices=["training", "inference"], help="Which features to create")
    parser.add_argument("--start-date", required=True, type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), help="Start date inclusive (YYYY-MM-DD)")
    parser.add_argument("--end-date",   required=True, type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), help="End date exclusive (YYYY-MM-DD)")
    parser.add_argument("--write-to-s3", action="store_true", help="Whether to write inference features to S3 (also enables S3 caching)")
    args = parser.parse_args()
    write_to_s3 = args.write_to_s3

    current = args.start_date
    while current < args.end_date:
        if args.features == "training":
            print(f"Creating training features for {current:%Y-%m-%d}")
            create_training_features_for_day(current)
        else:
            print(f"Creating inference features for {current:%Y-%m-%d}")
            create_inference_features_for_day(current, write_to_s3=write_to_s3)
        current += timedelta(days=1)