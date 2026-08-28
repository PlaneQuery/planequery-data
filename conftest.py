def pytest_addoption(parser):
    parser.addoption(
        "--label",
        action="store",
        default=None,
        help="Append a label to the PIA test artifact directory name.",
    )
