import os
from datetime import date
from pathlib import Path

os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[2]))

from data_profiling.cli import find_outliers


def test_find_outliers() -> None:
    counts = {
        date(2026, 1, 1): 98,
        date(2026, 1, 2): 100,
        date(2026, 1, 3): 102,
        date(2026, 1, 4): 0,
    }

    outliers, lower, upper = find_outliers(counts, sigma=2.0)

    assert outliers == [(date(2026, 1, 4), 0)]
    assert lower == 96
    assert upper == 104
