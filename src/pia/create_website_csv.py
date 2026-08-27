import polars as pl

from data_engineering.filters import pia_or_american_ladd
from data_engineering.utils import OUTPUT_DIR
from pia.ground_truth import (
    get_ground_truth_mappings,
    get_unique_pia_icaos_from_adsblol,
)


def create_pia_mappings_dataframe(
    all_pia_icaos: pl.Series, df_groundtruth: pl.DataFrame
) -> pl.DataFrame:
    """Create website mappings while retaining every known PIA ICAO."""
    all_pia_icao_set = set(all_pia_icaos)
    groundtruth_pia_icao_set = set(df_groundtruth.get_column("pia_icao"))
    missing_icaos = sorted(groundtruth_pia_icao_set - all_pia_icao_set)
    assert not missing_icaos, (
        "Ground-truth PIA ICAOs are missing from all_pia_icaos: "
        f"{missing_icaos[:20]}"
    )

    return pl.DataFrame({"icao": all_pia_icaos}).join(
        df_groundtruth.select("pia_icao", "real_icao", "registration"),
        left_on="icao",
        right_on="pia_icao",
        how="left",
    )


def main():
    all_pia_icaos = get_unique_pia_icaos_from_adsblol()
    print(f"Found {len(all_pia_icaos)} unique PIA ICAOs in ADS-B data.")
    df_groundtruth = get_ground_truth_mappings()
    print(f"Found {len(df_groundtruth)} unique PIA ICAOs in ground truth mappings.")

    df = create_pia_mappings_dataframe(all_pia_icaos, df_groundtruth)
    output_path = OUTPUT_DIR / "data" / "pia" / "pia_mappings.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    print(f"Wrote {len(df)} PIA mappings to {output_path}.")


if __name__ == "__main__":
    main()
