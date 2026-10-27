"""Provider days that must not be used to build complete flight outputs."""

from datetime import date


ADSBLOL_SKIP_DAYS = {
    # Published split tar is malformed; both parts match GitHub SHA-256 digests.
    # https://github.com/adsblol/globe_history_2026/releases/tag/v2026.04.30-planes-readsb-prod-0
    date(2026, 4, 30),
    date(2026, 5, 5),
    date(2026, 5, 6),
}
