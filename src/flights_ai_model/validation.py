import polars as pl
from datetime import date, datetime
import math
from data_engineering.adsb.read_adsb import read_adsb
MATCHED_DURATION = pl.duration(minutes=30)
from data_engineering.flights.flights_and_adsb_match import get_icaos_with_perfect_match
from data_engineering.bts.read_bts_ontime_to_flights import get_bts_flights_for_day
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day

def _airport_ident_for_match_expr(col: str) -> pl.Expr:
    ident = pl.col(col).str.to_uppercase()
    return (
        pl.when((ident.str.len_chars() == 4) & (ident.str.slice(0, 1) == "K"))
        .then(ident.str.slice(1))
        .otherwise(ident)
    )

def flights_df_comparison(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    left_on: str = "icao",
    right_on: str = "icao",
) -> tuple:
    """Compare two flights DataFrames using ICAO and time-proximity matching.

    For each ICAO present in both DataFrames, finds pairs of flights where both
    takeoff_time and landing_time are within 30 minutes of each other. # TODO: maybe this can be slightly diffrent.
    Each row can only be matched once (greedy bipartite matching sorted by
    smallest combined time difference first).

    Returns (df_all, stats) where df_all contains every row from both DataFrames
    with a 'match_status' column ('matched', 'df1_only', 'df2_only').  Matched
    rows have both sides populated; unmatched rows have nulls for the opposite
    side's columns (df2 columns are suffixed with '_df2').
    """
    # Snapshot column lists before we add internal tracking indices.
    df2_orig_cols = df2.columns

    df1 = df1.with_row_index("_idx1")
    df2 = df2.with_row_index("_idx2")

    # Use all rows from both DataFrames as candidates.
    common_icaos = set(df1[left_on].to_list()) & set(df2[right_on].to_list())


    # Inner join on ICAO, filter to pairs within ±30 min on both times.
    candidates = (
        df1
        .join(df2, left_on=left_on, right_on=right_on, how="inner", suffix="_df2")
        .with_columns(
            (pl.col("takeoff_time") - pl.col("takeoff_time_df2")).abs().alias("takeoff_time_diff"),
            (pl.col("landing_time") - pl.col("landing_time_df2")).abs().alias("landing_time_diff"),
        )
        .filter(
            (pl.col("takeoff_time_diff") <= MATCHED_DURATION) &
            (pl.col("landing_time_diff") <= MATCHED_DURATION)
        )
        .with_columns(
            (pl.col("takeoff_time_diff") + pl.col("landing_time_diff")).alias("_total_diff")
        )
        .sort("_total_diff")
        .with_row_index("_cand_idx")
    )

    # Greedy bipartite matching (best pair first, each row used at most once).
    used_idx1: set[int] = set()
    used_idx2: set[int] = set()
    matched_cand_indices: list[int] = []

    for row in candidates.iter_rows(named=True):
        i1, i2 = row["_idx1"], row["_idx2"]
        if i1 not in used_idx1 and i2 not in used_idx2:
            used_idx1.add(i1)
            used_idx2.add(i2)
            matched_cand_indices.append(row["_cand_idx"])

    df_good = candidates.filter(pl.col("_cand_idx").is_in(matched_cand_indices))

    # ── Build the full output DataFrame ────────────────────────────────────

    # Determine how df2 original columns are named inside the joined result.
    # The inner join drops df2's key column; non-key columns conflicting with
    # df1 columns get the "_df2" suffix.
    df1_col_set = set(df1.columns)  # includes _idx1
    df2_joined_names: dict[str, str] = {}  # orig df2 col -> name in join result
    for col in df2_orig_cols:
        if col == right_on:
            continue  # dropped by the join
        df2_joined_names[col] = f"{col}_df2" if col in df1_col_set else col

    # Reference schema: matched rows, minus internal helper columns.
    _internal = {"_idx1", "_idx2", "_cand_idx", "_total_diff"}
    df_matched_out = (
        df_good
        .drop([c for c in _internal if c in df_good.columns])
        .with_columns(pl.lit("matched").alias("match_status"))
    )
    output_cols = df_matched_out.columns  # canonical column order for concat

    def _fill_nulls(df: pl.DataFrame, ref: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
        """Add any columns from *cols* missing from *df* as nulls typed from *ref*."""
        for col in cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(ref[col].dtype).alias(col))
        return df

    # df1-only rows: all df1 rows that were not matched (any ICAO).
    df1_only = (
        df1
        .filter(~pl.col("_idx1").is_in(list(used_idx1)))
        .drop("_idx1")
        .with_columns(pl.lit("df1_only").alias("match_status"))
    )
    df1_only = _fill_nulls(df1_only, df_matched_out, output_cols).select(output_cols)

    # df2-only rows: all df2 rows that were not matched (any ICAO).
    # Rename df2 columns to match how they appear in the joined schema.
    rename_map = {orig: joined for orig, joined in df2_joined_names.items() if orig != joined}
    df2_only = (
        df2
        .filter(~pl.col("_idx2").is_in(list(used_idx2)))
        .drop("_idx2")
        .rename(rename_map)
        .with_columns(pl.lit("df2_only").alias("match_status"))
    )
    # If the join keys differ, rename df2's key column to match df1's key name.
    if right_on != left_on and right_on in df2_only.columns:
        df2_only = df2_only.rename({right_on: left_on})
    df2_only = _fill_nulls(df2_only, df_matched_out, output_cols).select(output_cols)

    df_all = pl.concat([df_matched_out, df1_only, df2_only])

    # ── Stats ───────────────────────────────────────────────────────────────
    n_correctly_matched   = len(matched_cand_indices)
    n_predicted           = len(df1)
    n_gold                = len(df2)
    n_predicted_unmatched = n_predicted - n_correctly_matched
    n_gold_missed         = n_gold - n_correctly_matched
    precision = n_correctly_matched / n_predicted if n_predicted > 0 else 0.0
    recall    = n_correctly_matched / n_gold       if n_gold > 0       else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ── Airport ident match on time-matched flights ──────────────────────────
    airport_stats: dict = {}
    for ident_col in ("takeoff_airport_ident", "landing_airport_ident"):
        df2_col = f"{ident_col}_df2"
        left_ident = _airport_ident_for_match_expr(ident_col)
        right_ident = _airport_ident_for_match_expr(df2_col)
        n_with_both = df_matched_out.filter(
            pl.col(ident_col).is_not_null() & pl.col(df2_col).is_not_null()
        ).height
        n_match = df_matched_out.filter(
            pl.col(ident_col).is_not_null() & pl.col(df2_col).is_not_null() &
            (left_ident == right_ident)
        ).height
        pct = n_match / n_with_both if n_with_both > 0 else float("nan")
        airport_stats[ident_col] = {"n_with_both": n_with_both, "n_match": n_match, "pct": pct}

    total_with_both = sum(s["n_with_both"] for s in airport_stats.values())
    total_match     = sum(s["n_match"]     for s in airport_stats.values())
    airport_stats["overall"] = {
        "n_with_both": total_with_both,
        "n_match": total_match,
        "pct": total_match / total_with_both if total_with_both > 0 else float("nan"),
    }

    print(f"\n--- Match Summary (ICAO-based, ±30 min window) ---")
    print(f"Common ICAOs: {len(common_icaos)}")
    print(f"Predicted flights: {n_predicted}  |  Gold flights: {n_gold}")
    print(f"Correctly matched: {n_correctly_matched}  |  Predicted unmatched: {n_predicted_unmatched}  |  Gold flights missed: {n_gold_missed}")
    print(f"Precision: {precision:.2%}  |  Recall: {recall:.2%}  |  F1: {f1:.2%}")
    if airport_stats:
        print(f"\n--- Airport Ident Agreement (on time-matched flights) ---")
        for ident_col, s in airport_stats.items():
            print(f"{ident_col}: {s['n_match']}/{s['n_with_both']} matched ({s['pct']:.2%})")

    stats = {
        "n_common_icaos": len(common_icaos),
        "n_predicted": n_predicted,
        "n_gold": n_gold,
        "n_correctly_matched": n_correctly_matched,
        "n_predicted_unmatched": n_predicted_unmatched,
        "n_gold_missed": n_gold_missed,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "airport_ident_pct_match": airport_stats["overall"]["pct"],
    }
    return df_all, stats

def process_icao(df_bts_icao: pl.DataFrame, df_adsb_icao: pl.DataFrame) -> bool | None: # TODO: Probalby delete? 
    """Returns True if any non-flight row is outside airport area, False if all within, None if insufficient data."""
    if df_adsb_icao.is_empty() or df_bts_icao.is_empty():
        return None

    df_s = (
        df_adsb_icao.select(["time", "baro_altitude_ft", "lat", "lon"])
        .sort("time").set_sorted("time").with_row_index("adsb_row_ix")
    )
    df_f = df_bts_icao.select(["icao", "takeoff_time", "landing_time"]).sort("takeoff_time")

    df_takeoff_match = (
        df_f.join_asof(df_s, left_on="takeoff_time", right_on="time", strategy="nearest")
        .select(pl.col("adsb_row_ix").alias("takeoff_adsb_row_ix"), pl.lit(True).alias("is_takeoff"))
    )
    df_landing_match = (
        df_f.sort("landing_time")
        .join_asof(df_s, left_on="landing_time", right_on="time", strategy="nearest")
        .select(pl.col("adsb_row_ix").alias("landing_adsb_row_ix"), pl.lit(True).alias("is_landing"))
    )

    df = (
        df_s
        .join(df_takeoff_match, left_on="adsb_row_ix", right_on="takeoff_adsb_row_ix", how="left")
        .join(df_landing_match, left_on="adsb_row_ix", right_on="landing_adsb_row_ix", how="left")
    )

    # in_flight=True between takeoff and landing (inclusive)
    df = df.with_columns(
        (
            pl.col("is_takeoff").fill_null(False).cast(pl.Int32).cum_sum() >
            pl.col("is_landing").fill_null(False).cast(pl.Int32).cum_sum().shift(1).fill_null(0)
        ).alias("in_flight")
    )

    df = df.with_columns([
        pl.when(pl.col("is_takeoff")).then(pl.col("lat")).otherwise(None).alias("next_takeoff_lat"),
        pl.when(pl.col("is_takeoff")).then(pl.col("lon")).otherwise(None).alias("next_takeoff_lon"),
        pl.when(pl.col("is_landing")).then(pl.col("lat")).otherwise(None).alias("last_landing_lat"),
        pl.when(pl.col("is_landing")).then(pl.col("lon")).otherwise(None).alias("last_landing_lon"),
    ])
    df = df.with_columns([
        pl.col("next_takeoff_lat").backward_fill(),
        pl.col("next_takeoff_lon").backward_fill(),
        pl.col("last_landing_lat").forward_fill(),
        pl.col("last_landing_lon").forward_fill(),
    ])

    df = df.filter(~pl.col("in_flight"))

    df = df.with_columns([
        pl.when(pl.col("last_landing_lat").is_not_null())
          .then(pl.col("last_landing_lat")).otherwise(pl.col("next_takeoff_lat")).alias("airport_lat"),
        pl.when(pl.col("last_landing_lon").is_not_null())
          .then(pl.col("last_landing_lon")).otherwise(pl.col("next_takeoff_lon")).alias("airport_lon"),
    ])

    # Drop rows where lat/lon or airport reference are null
    df = df.filter(
        pl.col("lat").is_not_null() & pl.col("lon").is_not_null() &
        pl.col("airport_lat").is_not_null() & pl.col("airport_lon").is_not_null()
    )
    if df.is_empty():
        return None

    df = df.with_columns(
        pl.struct(["lat", "lon", "airport_lat", "airport_lon"])
        .map_elements(
            lambda row: haversine(row["lat"], row["lon"], row["airport_lat"], row["airport_lon"]),
            return_dtype=pl.Float64,
        )
        .alias("distance_from_airport_km")
    )
    return (df["distance_from_airport_km"] > MAX_AIRPORT_DISTANCE_KM).any()


MAX_AIRPORT_DISTANCE_KM = 16 # in km
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface specified in decimal degrees.
    """
    R = 6371.0  # Earth radius in kilometers

    # Convert decimal degrees to radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(delta_lambda / 2.0) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in kilometers
def run_validation(current_date: datetime):
    pass

def get_validation_flights(current_date: date, df_adsb: pl.DataFrame): # this takes in current_date so probably not good function. May need run_validation. 
    if current_date < date(2026,1,1):
        df_val_flights = get_bts_flights_for_day(current_date, use_cache=True)
    elif current_date > date(2026,1,20):
        df_val_flights = get_sfdps_flights_day(current_date)
    else:
        raise ValueError("No validation data available for this date.")
    icaos = get_icaos_with_perfect_match(df_val_flights, df_adsb)
    df_val_flights = df_val_flights.filter(pl.col("icao").is_in(icaos))
    return df_val_flights
