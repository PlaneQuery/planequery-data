"""Shared ICAO bucketing used by canonical ADS-B parquet providers."""

from collections.abc import Sequence


ICAO_BUCKET_COUNT = 8
ICAO_BUCKET_SIZE = 0x1000000 // ICAO_BUCKET_COUNT


def icao_bucket(icao: str) -> int:
    if icao.startswith("~"):
        return ICAO_BUCKET_COUNT - 1
    icao_int = int(icao.lower(), 16)
    return min(icao_int // ICAO_BUCKET_SIZE, ICAO_BUCKET_COUNT - 1)


def query_icao_buckets(icaos: Sequence[str] | None) -> list[int] | None:
    if icaos is None:
        return None
    try:
        return sorted({icao_bucket(icao) for icao in icaos})
    except (AttributeError, TypeError, ValueError):
        # Preserve malformed/nonstandard identifiers by scanning all buckets.
        return None
