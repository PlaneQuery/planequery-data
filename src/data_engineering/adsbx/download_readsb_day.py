#!/usr/bin/env python3
"""Download ADSBExchange readsb-hist snapshots for a specific UTC day."""

import argparse
import concurrent.futures
import datetime as dt
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

from data_engineering.utils import OUTPUT_DIR


BASE_URL = "https://samples.adsbexchange.com/readsb-hist"
DEFAULT_OUT_DIR = OUTPUT_DIR / "data" / "raw" / "adsb-exchange" / "readsb-hist"


def iter_times(step_seconds: int):
	current = dt.datetime(2000, 1, 1)
	end = current + dt.timedelta(days=1)
	while current < end:
		yield current.strftime("%H%M%SZ")
		current += dt.timedelta(seconds=step_seconds)


def validate_gzip_json(path: pathlib.Path) -> bool:
	try:
		if path.stat().st_size == 0:
			return False
		with path.open("rb") as handle:
			return handle.read(2) == b"\x1f\x8b"
	except OSError:
		return False


def download_one(url: str, target: pathlib.Path, timeout: int, retries: int):
	if target.exists() and target.stat().st_size > 0 and validate_gzip_json(target):
		return "skipped", url

	tmp_target = target.with_suffix(target.suffix + ".part")
	headers = {
		"Accept-Encoding": "gzip",
		"User-Agent": "planequery-adsbx-downloader/1.0",
	}
	request = urllib.request.Request(url, headers=headers)

	for attempt in range(1, retries + 2):
		try:
			with urllib.request.urlopen(request, timeout=timeout) as response:
				status = getattr(response, "status", 200)
				if status != 200:
					raise urllib.error.HTTPError(
						url,
						status,
						"unexpected status",
						response.headers,
						None,
					)

				tmp_target.parent.mkdir(parents=True, exist_ok=True)
				with tmp_target.open("wb") as output:
					while True:
						chunk = response.read(1024 * 256)
						if not chunk:
							break
						output.write(chunk)

			if not validate_gzip_json(tmp_target):
				raise OSError("downloaded file is not valid gzip JSON")

			os.replace(tmp_target, target)
			return "downloaded", url
		except Exception as exc:  # noqa: BLE001
			if tmp_target.exists():
				tmp_target.unlink()
			if attempt > retries:
				return "failed", f"{url} :: {exc}"
			time.sleep(min(30, 2**attempt))

	return "failed", f"{url} :: unexpected error"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Download ADSBExchange readsb-hist snapshots for one UTC day.",
	)
	parser.add_argument("date", help="UTC date in YYYY-MM-DD format")
	parser.add_argument(
		"--out-dir",
		type=pathlib.Path,
		default=DEFAULT_OUT_DIR,
		help=(
			"Base output directory, default: "
			f"{DEFAULT_OUT_DIR}"
		),
	)
	parser.add_argument("--step-seconds", type=int, default=5)
	parser.add_argument("--workers", type=int, default=12)
	parser.add_argument("--timeout", type=int, default=60)
	parser.add_argument("--retries", type=int, default=3)
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	day = dt.date.fromisoformat(args.date)
	year = f"{day.year:04d}"
	month = f"{day.month:02d}"
	day_str = f"{day.day:02d}"

	out_dir = args.out_dir / year / month / day_str
	tasks = []
	for label in iter_times(args.step_seconds):
		filename = f"{label}.json.gz"
		url = f"{BASE_URL}/{year}/{month}/{day_str}/{filename}"
		tasks.append((url, out_dir / filename))

	print(f"Downloading {len(tasks)} files to {out_dir}", flush=True)
	counts = {"downloaded": 0, "skipped": 0, "failed": 0}
	failures = []
	started = time.time()

	with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
		futures = [
			executor.submit(download_one, url, target, args.timeout, args.retries)
			for url, target in tasks
		]
		for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
			status, detail = future.result()
			counts[status] += 1
			if status == "failed":
				failures.append(detail)
			if index % 250 == 0 or index == len(futures):
				elapsed = time.time() - started
				print(
					f"{index}/{len(futures)} complete "
					f"(downloaded={counts['downloaded']}, skipped={counts['skipped']}, "
					f"failed={counts['failed']}, elapsed={elapsed:.0f}s)",
					flush=True,
				)

	if failures:
		failure_path = out_dir / "failed-downloads.txt"
		failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
		print(f"Wrote {len(failures)} failures to {failure_path}", file=sys.stderr)
		return 1

	print(
		f"Done: downloaded={counts['downloaded']}, skipped={counts['skipped']}, failed=0",
		flush=True,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
