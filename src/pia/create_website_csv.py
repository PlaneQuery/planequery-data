import polars as pl

from data_engineering.utils import OUTPUT_DIR
from pia.ground_truth import (
    get_ground_truth_mappings,
    get_unique_pia_icaos_from_adsblol,
)


def create_pia_mappings_dataframe(
    all_pia_icaos: pl.Series, df_groundtruth: pl.DataFrame
) -> pl.DataFrame:
    """Create website mappings, retaining ICAOs without a ground-truth match."""
    missing_icaos = sorted(
        set(df_groundtruth.get_column("pia_icao")) - set(all_pia_icaos)
    )
    print(
        "Ground-truth PIA ICAOs are missing from all_pia_icaos: "
        f"{missing_icaos}"
    )

    return pl.DataFrame({"icao": all_pia_icaos}).join(
        df_groundtruth.select("pia_icao", "real_icao", "registration"),
        left_on="icao",
        right_on="pia_icao",
        how="left",
    )


def main(use_cache: bool = True):
    all_pia_icaos = get_unique_pia_icaos_from_adsblol(use_cache=use_cache)
    print(f"Found {len(all_pia_icaos)} unique PIA ICAOs in ADS-B data.")
    df_groundtruth = get_ground_truth_mappings(use_cache=use_cache)
    print(f"Found {len(df_groundtruth)} unique PIA ICAOs in ground truth mappings.")

    df = create_pia_mappings_dataframe(all_pia_icaos, df_groundtruth)
    output_path = OUTPUT_DIR / "data" / "pia" / "pia_mappings.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    print(f"Wrote {len(df)} PIA mappings to {output_path}.")


if __name__ == "__main__":
    main()
