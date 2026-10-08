from datetime import date


def pytest_addoption(parser):
    parser.addoption(
        "--label",
        action="store",
        default=None,
        help="Append a label to the PIA test artifact directory name.",
    )
    parser.addoption(
        "--no-refresh-cache",
        action="store_false",
        dest="refresh_cache",
        default=True,
        help=(
            "Reuse valid flight and evaluation cache entries, generating "
            "missing entries as needed."
        ),
    )
    parser.addoption(
        "--test-source",
        nargs="+",
        default=None,
        help="Run quality baselines for only these test sources.",
    )
    parser.addoption(
        "--gold-source",
        nargs="+",
        default=None,
        help="Run quality baselines for only these gold sources.",
    )
    parser.addoption(
        "--adsb-source",
        nargs="+",
        choices=("adsblol", "adsbx", "opensky"),
        default=None,
        help=(
            "Run only matching ADS-B-dependent quality baselines; baselines "
            "whose test source does not use ADS-B are always included."
        ),
    )
    parser.addoption(
        "--test-dates",
        nargs="+",
        type=date.fromisoformat,
        default=None,
        help="Run quality baselines for only these UTC flight dates.",
    )
