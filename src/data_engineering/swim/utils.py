from datetime import date, timedelta

def date_range(start: date, end: date) -> list[date]:
    """Inclusive date range."""
    return [
        start + timedelta(days=i)
        for i in range((end - start).days + 1)
    ]

MISSING_SFDPS_DAYS = [
    *date_range(date(2026, 1, 1), date(2026, 1, 19)),
    *date_range(date(2026, 4, 22), date(2026, 5, 8)),
    date(2026, 6, 9),
]

