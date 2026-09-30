"""Shared OpenSky Trino connection and Arrow helpers."""

import os
from typing import Any

import pyarrow as pa
from trino.auth import OAuth2Authentication
from trino.dbapi import Connection, connect


BATCH_SIZE = 100_000
DEFAULT_TRINO_USER = os.environ.get("OPENSKY_TRINO_USER", "goodejonah")


def rows_to_arrow(
    rows: list[tuple[Any, ...]],
    columns: list[str],
) -> pa.Table:
    """Convert one batch of Trino DB-API rows into an Arrow table."""
    if not rows:
        return pa.table({column: [] for column in columns})

    return pa.Table.from_arrays(
        [pa.array(values) for values in zip(*rows)],
        names=columns,
    )


def create_connection(user: str) -> Connection:
    """Create an external OAuth-authenticated OpenSky Trino connection."""
    return connect(
        host="trino.opensky-network.org",
        port=443,
        user=user.lower(),
        catalog="minio",
        schema="osky",
        http_scheme="https",
        auth=OAuth2Authentication(),
        timezone="UTC",
        encoding=["json+zstd", "json"],
    )
