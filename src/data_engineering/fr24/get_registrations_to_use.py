from datetime import date
import hashlib
import json
import math
import random
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl

TARGET_DATE = date(2026, 6, 1)
N_PLANES = 1000
RANDOM_SEED = 42
FR24_REGISTRATION_VALIDITY_VERSION = "v1"
FR24_REGISTRATION_REGEX = r"^[A-Z0-9][A-Z0-9-]{1,9}$"
FR24_REGISTRATION_PATTERN = re.compile(FR24_REGISTRATION_REGEX)
COUNTRIES_PATH = Path(
    "/Volumes/T2-SSD/planequery/data/raw/natural-earth/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
)

EUROPE_ISO3 = {
    "ALB", "AND", "AUT", "BEL", "BIH", "BGR", "BLR", "CHE", "CYP",
    "CZE", "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC",
    "HRV", "HUN", "IRL", "ISL", "ITA", "LIE", "LTU", "LUX", "LVA",
    "MCO", "MDA", "MKD", "MLT", "MNE", "NLD", "NOR", "POL", "PRT",
    "ROU", "SMR", "SRB", "SVK", "SVN", "SWE", "UKR", "VAT",
}


def is_valid_fr24_registration(registration: str | None) -> bool:
    if registration is None:
        return False
    return FR24_REGISTRATION_PATTERN.fullmatch(registration.strip().upper()) is not None


def proportional_country_quota(
    registration_country: pd.DataFrame,
    n: int,
) -> dict[str, int]:
    counts = registration_country.groupby("ISO_A3")["registration"].nunique()
    raw = counts / counts.sum() * n
    quota = raw.apply(math.floor).clip(lower=1)

    remaining = n - int(quota.sum())
    for country in (
        raw - raw.apply(math.floor)
    ).sort_values(ascending=False).index[:remaining]:
        quota[country] += 1

    return quota.astype(int).to_dict()


def _country_registration_maps(
    registration_country: pd.DataFrame,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    registration_to_countries = (
        registration_country
        .groupby("registration")["ISO_A3"]
        .agg(set)
        .to_dict()
    )
    country_to_registrations = (
        registration_country
        .groupby("ISO_A3")["registration"]
        .agg(set)
        .to_dict()
    )
    return registration_to_countries, country_to_registrations


def _registrations_cache_path(
    target_date: date,
    n_planes: int,
    seed: int,
    countries_path: Path | str,
) -> Path:
    from data_engineering.utils import OUTPUT_DIR

    countries_path_hash = hashlib.sha1(
        str(Path(countries_path)).encode()
    ).hexdigest()[:8]
    cache_filename = (
        f"fr24_registrations_{target_date:%Y-%m-%d}_"
        f"{n_planes}_{seed}_{FR24_REGISTRATION_VALIDITY_VERSION}_"
        f"{countries_path_hash}.json"
    )
    return (
        OUTPUT_DIR
        / "tmp"
        / cache_filename
    )


def _registration_country_for_date(
    target_date: date,
    countries_path: Path | str,
) -> pd.DataFrame:
    from data_engineering.adsb.read_adsb import read_adsb

    df = read_adsb(target_date)

    countries = gpd.read_file(Path(countries_path)).to_crs("EPSG:4326")
    europe = (
        countries[countries["ADM0_A3"].isin(EUROPE_ISO3)]
        [["ADM0_A3", "ADMIN", "geometry"]]
        .rename(columns={"ADM0_A3": "ISO_A3"})
    )

    minx, miny, maxx, maxy = europe.total_bounds

    adsb_points = (
        df
        .select(["registration", "lat", "lon"])
        .with_columns(
            pl.col("registration").cast(pl.String).str.strip_chars().str.to_uppercase(),
            pl.col("lat").cast(pl.Float64, strict=False),
            pl.col("lon").cast(pl.Float64, strict=False),
        )
        .drop_nulls(["registration", "lat", "lon"])
        .filter(pl.col("registration") != "")
        .filter(pl.col("registration").str.contains(FR24_REGISTRATION_REGEX))
        .filter(
            pl.col("lat").is_between(miny, maxy)
            & pl.col("lon").is_between(minx, maxx)
        )
        .unique()
        .to_pandas()
    )

    points = gpd.GeoDataFrame(
        adsb_points,
        geometry=gpd.points_from_xy(adsb_points["lon"], adsb_points["lat"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(points, europe, how="inner", predicate="intersects")
    return (
        joined[["registration", "ISO_A3"]]
        .dropna()
        .drop_duplicates()
    )


def _select_registrations(
    registration_country: pd.DataFrame,
    n: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    quota = proportional_country_quota(registration_country, n)
    selected: set[str] = set()
    registration_to_countries, country_to_registrations = _country_registration_maps(
        registration_country
    )

    # Guarantee country coverage first, beginning with countries that have fewer options.
    for country, regs in sorted(
        country_to_registrations.items(),
        key=lambda item: len(item[1]),
    ):
        if selected.intersection(regs):
            continue

        selected.add(
            max(
                regs,
                key=lambda reg: sum(
                    1 / len(country_to_registrations[country_seen])
                    for country_seen in registration_to_countries[reg]
                ),
            ),
        )

    if len(selected) > n:
        raise ValueError(
            f"Need at least {len(selected)} registrations to cover every country"
        )

    while len(selected) < n:
        selected_country_counts = (
            registration_country[registration_country["registration"].isin(selected)]
            .groupby("ISO_A3")["registration"]
            .nunique()
            .to_dict()
        )
        deficits = {
            country: quota[country] - selected_country_counts.get(country, 0)
            for country in quota
        }
        candidates = list(set(registration_to_countries) - selected)

        def score(registration: str) -> int:
            return sum(
                max(deficits.get(country, 0), 0)
                for country in registration_to_countries[registration]
            )

        best_score = max(score(registration) for registration in candidates)
        if best_score <= 0:
            selected.add(rng.choice(candidates))
            continue

        selected.add(
            rng.choice([
                registration
                for registration in candidates
                if score(registration) == best_score
            ])
        )

    return sorted(selected)


def get_registrations_to_use(
    target_date: date = TARGET_DATE,
    n_planes: int = N_PLANES,
    seed: int = RANDOM_SEED,
    countries_path: Path | str = COUNTRIES_PATH,
) -> list[str]:
    cache_path = _registrations_cache_path(target_date, n_planes, seed, countries_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    registration_country = _registration_country_for_date(target_date, countries_path)
    registration_to_countries, country_to_registrations = _country_registration_maps(
        registration_country
    )

    missing_countries = sorted(EUROPE_ISO3 - set(country_to_registrations))
    if missing_countries:
        raise ValueError(f"Cannot cover countries with no planes: {missing_countries}")

    if len(registration_to_countries) < n_planes:
        raise ValueError(
            f"Only {len(registration_to_countries)} unique registrations available"
        )

    registrations = _select_registrations(
        registration_country=registration_country,
        n=n_planes,
        seed=seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(registrations, indent=2))
    return registrations


if __name__ == "__main__":
    regs = get_registrations_to_use(date(2026, 6, 1))
    print(len(regs))
    print(regs[0:10])
