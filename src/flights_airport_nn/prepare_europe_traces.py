"""Prepare explicitly cohort-scoped historical full traces for European evaluation.

The output is deliberately outside the ordinary full-network cache. Missing
aircraft remain in the reference population; HTTP failures abort preparation.
"""

import argparse
import hashlib
import json
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import polars as pl

from data_engineering.adsb.trace_files_to_parquet import process_trace_files
from data_engineering.eurocontrol.read import eurocontrol_flights_parquet_path


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    subprocess.run(
        [
            "curl",
            "-fsSL",
            "--retry",
            "3",
            "--max-time",
            "180",
            "-o",
            str(temporary),
            url,
        ],
        check=True,
    )
    temporary.replace(path)


def prepare(day, source, root):
    reference_path = eurocontrol_flights_parquet_path(day)
    aircraft = set(
        pl.read_parquet(reference_path)["icao"].drop_nulls().str.to_lowercase()
    )
    trace_dir = root / "traces" / source / f"{day:%Y/%m/%d}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / f"acquisition-{source}-{day}.json"
    urls = []
    if source == "adsbx":
        base = f"https://samples.adsbexchange.com/traces/{day:%Y/%m/%d}/"
        index = root / "downloads" / f"adsbx-{day}-index.json"
        download(base + "index.json", index)
        available = set(json.loads(index.read_text())["traces"])
        selected = sorted(aircraft & available)

        def fetch(icao):
            download(
                base + f"{icao[-2:]}/trace_full_{icao}.json",
                trace_dir / f"trace_full_{icao}.json",
            )

        with ThreadPoolExecutor(max_workers=16) as pool:
            for i, _ in enumerate(pool.map(fetch, selected), 1):
                if i % 500 == 0:
                    print(f"{source} {day}: downloaded {i}/{len(selected)}", flush=True)
        urls = [base + "index.json", base + "{last2}/trace_full_{icao}.json"]
    else:
        tag = f"v{day:%Y.%m.%d}-planes-readsb-prod-0"
        parts = [
            root / "downloads" / f"june{day.day:02d}.tar.{part}"
            for part in ("aa", "ab")
        ]
        # Assets are downloaded first; concatenation is streamed, never extracted
        # by archive paths. Only regular whitelisted trace basenames are copied.
        if any(not p.exists() for p in parts):
            raise FileNotFoundError(parts)
        selected = set()
        with subprocess.Popen(
            ["cat", *map(str, parts)], stdout=subprocess.PIPE
        ) as stream:
            with tarfile.open(fileobj=stream.stdout, mode="r|") as archive:
                for member in archive:
                    name = Path(member.name).name
                    if (
                        not member.isfile()
                        or not name.startswith("trace_full_")
                        or not name.endswith(".json")
                    ):
                        continue
                    icao = name[len("trace_full_") : -len(".json")].lower()
                    if icao in aircraft:
                        selected.add(icao)
                        target = trace_dir / name
                        if not target.exists():
                            target.write_bytes(archive.extractfile(member).read())
            if stream.wait() != 0:
                raise RuntimeError("Archive concatenation failed")
        urls = [
            f"https://github.com/adsblol/globe_history_{day.year}/releases/download/{tag}/{tag}.tar.{part}"
            for part in ("aa", "ab")
        ]
    output = root / "parquet" / source
    marker = output / f"complete-{day}.json"
    if not marker.exists():
        rows = process_trace_files(
            trace_dir, output_root=output, max_workers=4, submit_chunk_size=16
        )
        marker.write_text(json.dumps({"rows": rows}) + "\n")
    manifest_path.write_text(
        json.dumps(
            {
                "date": str(day),
                "source": source,
                "representation": "full traces",
                "scope": "Only aircraft in the daily EUROCONTROL reference; absence is retained in evaluation",
                "reference": str(reference_path),
                "reference_sha256": hashlib.sha256(
                    reference_path.read_bytes()
                ).hexdigest(),
                "reference_aircraft": sorted(aircraft),
                "available_aircraft": sorted(selected),
                "absent_aircraft": sorted(aircraft - set(selected)),
                "urls": urls,
                "parquet_root": str(output),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Prepared {source} {day}: {len(selected)}/{len(aircraft)} reference aircraft",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--source", choices=["adsblol", "adsbx"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.date, args.source, args.root)
