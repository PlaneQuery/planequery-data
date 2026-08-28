from datetime import date
from pathlib import Path

import polars as pl

from data_engineering.ground_control import load_ground_control_mappings
from data_engineering.openairframes.read import (
    get_openairframes_faa_df,
    scan_latest_faa_registration_icao_for_join,
)
from data_engineering.utils import OUTPUT_DIR
from pia.identify_ladd_planes.pia_broadcast_adsb import get_pia_to_reg_broadcast_icaos
from pia.identify_ladd_planes.pia_callsign_switch import get_pia_callsign_switch_icaos

OUTPUT_PATH = OUTPUT_DIR / "cache/pia"
ALL_PIA_ICAOS_CACHE_PATH = Path(
    "resources/pia/all_pias_observed_2024-01-01_2026-08-26.csv"
)
GRAPH_SOLVE_PATH = Path("resources/pia/graph_solve.csv")
MAPPING_COLUMNS = ["pia_icao", "registration", "real_icao"]
MAPPING_SOURCE_PRIORITIES = {
    "graph_solve": 0,
    "pia_broadcast_adsb": 1,
    "pia_callsign_switch": 2,
    "ground_control": 3,
}
MAPPING_SOURCE_LABELS = {
    "graph_solve": "graph-solve",
    "ground_control": "groundcontrol",
    "pia_broadcast_adsb": "callsign",
    "pia_callsign_switch": "callsign",
}


def load_graph_solve_mappings() -> pl.DataFrame:
    """Load graph-solved mappings and add registrations from FAA data."""
    graph_mappings = (
        pl.scan_csv(
            GRAPH_SOLVE_PATH,
            schema_overrides={"pia_icao": pl.String, "real_icao": pl.String},
        )
        .with_columns(
            pl.col("pia_icao").str.strip_chars().str.to_lowercase(),
            pl.col("real_icao").str.strip_chars().str.to_lowercase(),
        )
        .filter(
            pl.col("pia_icao").is_not_null()
            & (pl.col("pia_icao") != "")
            & pl.col("real_icao").is_not_null()
            & (pl.col("real_icao") != "")
        )
        .unique()
    )
    registrations = (
        scan_latest_faa_registration_icao_for_join()
        .select("real_icao", "registration")
        .unique(subset=["real_icao"], keep="last")
    )
    mappings = graph_mappings.join(
        registrations,
        on="real_icao",
        how="left",
    ).collect()
    unmatched = mappings.filter(pl.col("registration").is_null())
    if not unmatched.is_empty():
        print(
            f"{unmatched.height} graph-solve mappings did not match an FAA "
            f"registration: {unmatched.get_column('real_icao').to_list()}"
        )
    return mappings.select(MAPPING_COLUMNS)


def _mapping_source_df(df: pl.DataFrame, source: str) -> pl.DataFrame:
    return (
        df.select(MAPPING_COLUMNS)
        .with_columns(
            pl.col("pia_icao").str.strip_chars().str.to_lowercase(),
            pl.col("registration").str.strip_chars().str.to_uppercase(),
            pl.col("real_icao").str.strip_chars().str.to_lowercase(),
            pl.lit(source).alias("mapping_source"),
            pl.lit(MAPPING_SOURCE_PRIORITIES[source]).alias("source_priority"),
        )
        .filter(
            pl.col("pia_icao").is_not_null()
            & (pl.col("pia_icao") != "")
            & pl.col("registration").is_not_null()
            & (pl.col("registration") != "")
            & pl.col("real_icao").is_not_null()
            & (pl.col("real_icao") != "")
        )
        .unique()
    )


def find_cross_source_mapping_disagreements(
    mappings: pl.DataFrame,
) -> pl.DataFrame:
    """Return mappings where two or more sources disagree for one PIA ICAO."""
    conflicting_icaos = (
        mappings.group_by("pia_icao")
        .agg(
            pl.struct("registration", "real_icao")
            .n_unique()
            .alias("mapping_count"),
            pl.col("mapping_source").n_unique().alias("source_count"),
        )
        .filter((pl.col("mapping_count") > 1) & (pl.col("source_count") > 1))
        .select("pia_icao")
    )
    return (
        mappings.join(conflicting_icaos, on="pia_icao", how="inner")
        .select("pia_icao", "mapping_source", "registration", "real_icao")
        .unique()
        .sort(["pia_icao", "mapping_source", "real_icao"])
    )


def get_ground_truth_mappings(
    use_cache: bool = True,
    start_date: date = date(2024, 1, 1),
) -> pl.DataFrame:
    start_date_str = start_date.isoformat()
    output_path = OUTPUT_PATH / f"ground_truth_mappings_start_date={start_date_str}.parquet"
    df_0 = (
        load_ground_control_mappings()
        .filter(pl.col("date") > start_date)
        .drop(["date"])
    )
    df_1 = get_pia_to_reg_broadcast_icaos(use_cache=use_cache)
    df_2 = get_pia_callsign_switch_icaos(use_cache=use_cache)
    df_graph_solve = load_graph_solve_mappings()
    other_source_icaos = (
        set(df_graph_solve.get_column("pia_icao"))
        | set(df_1.get_column("pia_icao"))
        | set(df_2.get_column("pia_icao"))
    )
    ground_control_only = set(df_0.get_column("pia_icao")) - other_source_icaos
    print(f"Mappings found only in ground control: {len(ground_control_only)}")
    mappings = pl.concat(
        [
            _mapping_source_df(df_graph_solve, "graph_solve"),
            _mapping_source_df(df_0, "ground_control"),
            _mapping_source_df(df_1, "pia_broadcast_adsb"),
            _mapping_source_df(df_2, "pia_callsign_switch"),
        ],
        how="vertical",
    )
    disagreements = find_cross_source_mapping_disagreements(mappings)
    if not disagreements.is_empty():
        print(
            "Mapping sources disagree for these PIA ICAOs: "
            f"{disagreements.to_dicts()}"
        )

    df = (
        mappings.sort(["source_priority", "pia_icao", "real_icao"])
        .unique(subset=["pia_icao"], keep="first", maintain_order=True)
        .with_columns(
            pl.col("mapping_source")
            .replace_strict(MAPPING_SOURCE_LABELS)
            .alias("source")
        )
        .select(*MAPPING_COLUMNS, "source")
    )
    if not use_cache:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)
    return df


BASE_PATH = OUTPUT_DIR / "data/raw/adsblol/parquet_output/v6"


def get_unique_pia_icaos_from_adsblol(use_cache: bool = True) -> pl.Series:
    """Return observed PIA ICAOs, using the resource CSV when cached."""
    if use_cache:
        pia_icaos = (
            pl.read_csv(
                ALL_PIA_ICAOS_CACHE_PATH,
                schema_overrides={"icao": pl.String},
            )
            .get_column("icao")
            .str.strip_chars()
            .str.to_lowercase()
            .unique()
            .sort()
        )
    else:
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


if __name__ == "__main__":
    print(get_ground_truth_mappings(use_cache=False))
