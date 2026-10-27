"""Download all reporting-carrier on-time fields, partitioned by month.

Example::

    python -m data_engineering.bts.download_ontime 2025-01-01 2026-01-01

Dates select whole months overlapping [start_date, end_date); rows are not
trimmed to the supplied days. Existing files missing fields are refreshed.
"""

import argparse
import csv
import re
import shutil
import tempfile
import zipfile
from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path

import requests
from lxml import html
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data_engineering.utils import OUTPUT_DIR


DOWNLOAD_URL = (
    "https://www.transtats.bts.gov/DL_SelectFields.aspx"
    "?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr"
)
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "data" / "raw" / "bts" / "v2"
TIMEOUT = (30, 600)


def months_in_range(start_date: date, end_date: date) -> Iterator[date]:
    """Yield first-of-month dates for every month touched by the interval."""
    if end_date <= start_date:
        raise ValueError("END_DATE must be after START_DATE")
    month = start_date.replace(day=1)
    while month < end_date:
        yield month
        month = date(month.year + month.month // 12, month.month % 12 + 1, 1)


def download_form(page: str, month: date) -> tuple[dict[str, str], set[str]]:
    """Retain ASP.NET state and select every data-field checkbox."""
    tree = html.fromstring(page)
    forms = tree.xpath('//form[@id="form1"]')
    if len(forms) != 1:
        raise ValueError("BTS did not return its download form")
    form = forms[0]
    fields = {
        node.get("name"): node.get("value", "")
        for node in form.xpath('.//input[@type="hidden"][@name]')
    }
    columns = {
        node.get("name")
        for node in form.xpath('.//input[@type="checkbox"][@name]')
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", node.get("name"))
    }
    if not {"YEAR", "MONTH", "FL_DATE", "ARR_DELAY"} <= columns:
        raise ValueError("BTS form is missing expected on-time fields")
    for name, value in (("cboYear", str(month.year)), ("cboPeriod", str(month.month))):
        available = form.xpath(f'.//select[@name="{name}"]/option/@value')
        if value not in available:
            raise ValueError(f"BTS does not offer {name}={value}")
    fields.update({column: "on" for column in columns})
    fields.update(
        cboGeography="All", cboYear=str(month.year),
        cboPeriod=str(month.month), btnDownload="Download",
    )
    return fields, columns


def validate_csv(path: Path, month: date, columns: set[str]) -> int:
    """Reject missing fields, empty exports, malformed rows, or wrong months."""
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing BTS columns: {', '.join(sorted(missing))}")
        count = 0
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed BTS CSV row")
            if int(row["YEAR"]) != month.year or int(row["MONTH"]) != month.month:
                raise ValueError(f"BTS returned data outside {month:%Y-%m}")
            count += 1
        if not count:
            raise ValueError(f"BTS returned no rows for {month:%Y-%m}")
        return count


def create_session() -> requests.Session:
    session = requests.Session()
    # POST only requests an export; retrying it does not mutate source data.
    retry = Retry(
        total=3, backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_month(
    month: date,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
    session: requests.Session | None = None,
) -> Path:
    """Download and validate all fields before atomically replacing a CSV."""
    if session is None:
        with create_session() as owned_session:
            return download_month(
                month, output_dir=output_dir, overwrite=overwrite,
                session=owned_session,
            )
    output = (
        output_dir / f"year={month.year}" / f"month={month.month:02d}"
        / "T_ONTIME_REPORTING.csv"
    )
    # GET and POST share cookies: without the session BTS returns HTML, not ZIP.
    with session.get(DOWNLOAD_URL, timeout=TIMEOUT) as response:
        response.raise_for_status()
        fields, columns = download_form(response.text, month)
    if output.exists() and not overwrite:
        try:
            validate_csv(output, month, columns)
        except (ValueError, csv.Error, UnicodeError):
            print(f"Refreshing {month:%Y-%m}: existing CSV is incomplete")
        else:
            print(f"Skipping {month:%Y-%m}: all {len(columns)} fields already downloaded")
            return output

    print(f"Downloading {month:%Y-%m}: all {len(columns)} fields", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bts-", dir=output.parent) as temp:
        archive = Path(temp) / "download.zip"
        with session.post(
            DOWNLOAD_URL, data=fields, stream=True, timeout=TIMEOUT,
        ) as response:
            response.raise_for_status()
            with archive.open("wb") as target:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    target.write(chunk)
        if not zipfile.is_zipfile(archive):
            raise ValueError(f"BTS did not return a ZIP for {month:%Y-%m}; data may be unavailable")
        extracted = Path(temp) / "data.csv"
        with zipfile.ZipFile(archive) as zipped:
            members = [n for n in zipped.namelist() if n.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"Expected one BTS CSV, received {len(members)}")
            # Stream the sole CSV to our own path; never trust archive paths.
            with zipped.open(members[0]) as source, extracted.open("wb") as target:
                shutil.copyfileobj(source, target)
        rows = validate_csv(extracted, month, columns)
        extracted.replace(output)
    print(f"Saved {rows:,} rows -> {output}", flush=True)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(
        "Download all BTS reporting-carrier on-time fields for every full month "
        "overlapping [START_DATE, END_DATE)."
    ))
    parser.add_argument("start_date", type=date.fromisoformat, help="Inclusive start (YYYY-MM-DD)")
    parser.add_argument("end_date", type=date.fromisoformat, help="Exclusive end (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"BTS output root (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redownload even when an existing CSV has all fields")
    args = parser.parse_args(argv)
    if args.end_date <= args.start_date:
        parser.error("END_DATE must be after START_DATE")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    with create_session() as session:
        for month in months_in_range(args.start_date, args.end_date):
            download_month(month, output_dir=args.output_dir,
                           overwrite=args.overwrite, session=session)


if __name__ == "__main__":
    main()
