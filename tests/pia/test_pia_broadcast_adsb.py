import os
from pathlib import Path

import polars as pl

os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[2]))

from pia.identify_ladd_planes.pia_broadcast_adsb import (
    map_pia_n_numbers_to_real_icaos,
)


def test_maps_only_unambiguous_pia_tail_numbers_to_faa_icao():
    pia_messages = pl.DataFrame(
        {
            "icao": ["pia001", "pia002", "pia002", "pia003", "pia004", "pia005"],
            "callsign": [" n100 ", "N200", "N201", "N300", "N400", "N500"],
        }
    )
    faa = pl.DataFrame(
        {
            "registration_number": ["N100", "N200", "N201", "N400", "N500", "N500"],
            "transponder_code_hex": ["A0B001", "A0B002", "A0B003", None, "A0B005", "A0B006"],
        }
    )

    result = map_pia_n_numbers_to_real_icaos(pia_messages, faa)

    assert result.to_dicts() == [
        {"icao": "pia001", "callsign": "N100", "real_icao": "a0b001"}
    ]
