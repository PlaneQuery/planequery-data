#!/usr/bin/env python3
"""Download ADSBX readsb-hist snapshots for a specific UTC day."""

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
DEFAULT_OUT_DIR = OUTPUT_DIR / "data" / "raw" / "adsbx" / "readsb-hist"
DEFAULT_RETRY_PASSES = 3


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


def download_complete_path(out_dir: pathlib.Path, step_seconds: int = 5) -> pathlib.Path:
	return out_dir / f".download-complete-{step_seconds}s"


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


def download_tasks(
	tasks: list[tuple[str, pathlib.Path]],
	workers: int,
	timeout: int,
	retries: int,
	retry_passes: int,
) -> tuple[dict[str, int], list[str]]:
	"""Download tasks, then make delayed passes over only the failures."""
	counts = {"downloaded": 0, "skipped": 0, "failed": 0}
	pending = tasks
	failures: list[str] = []
	started = time.time()

	for pass_index in range(max(0, retry_passes) + 1):
		if pass_index:
			delay = min(120, 15 * 2 ** (pass_index - 1))
			print(
				f"Retry pass {pass_index}/{retry_passes} for "
				f"{len(pending)} failed download(s) in {delay}s",
				flush=True,
			)
			time.sleep(delay)

		next_pending: list[tuple[str, pathlib.Path]] = []
		failures = []
		with concurrent.futures.ThreadPoolExecutor(
			max_workers=min(workers, len(pending)),
		) as executor:
			future_tasks = {
				executor.submit(download_one, url, target, timeout, retries): (url, target)
				for url, target in pending
			}
			for index, future in enumerate(
				concurrent.futures.as_completed(future_tasks),
				start=1,
			):
				status, detail = future.result()
				if status == "failed":
					next_pending.append(future_tasks[future])
					failures.append(detail)
				else:
					counts[status] += 1
				if index % 250 == 0 or index == len(future_tasks):
					elapsed = time.time() - started
					print(
						f"Pass {pass_index + 1}: {index}/{len(future_tasks)} complete "
						f"(downloaded={counts['downloaded']}, "
						f"skipped={counts['skipped']}, "
						f"pending={len(next_pending)}, elapsed={elapsed:.0f}s)",
						flush=True,
					)

		pending = next_pending
		if not pending:
			break

	counts["failed"] = len(pending)
	return counts, failures


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Download ADSBX readsb-hist snapshots for one UTC day.",
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
	parser.add_argument(
		"--retry-passes",
		type=int,
		default=DEFAULT_RETRY_PASSES,
		help=(
			"Additional delayed passes over only the downloads that still failed; "
			f"default: {DEFAULT_RETRY_PASSES}"
		),
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	day = dt.date.fromisoformat(args.date)
	year = f"{day.year:04d}"
	month = f"{day.month:02d}"
	day_str = f"{day.day:02d}"

	out_dir = args.out_dir / year / month / day_str
	out_dir.mkdir(parents=True, exist_ok=True)
	completion_path = download_complete_path(out_dir, args.step_seconds)
	completion_path.unlink(missing_ok=True)
	tasks = []
	for label in iter_times(args.step_seconds):
		filename = f"{label}.json.gz"
		url = f"{BASE_URL}/{year}/{month}/{day_str}/{filename}"
		tasks.append((url, out_dir / filename))

	print(f"Downloading {len(tasks)} files to {out_dir}", flush=True)
	counts, failures = download_tasks(
		tasks,
		workers=args.workers,
		timeout=args.timeout,
		retries=args.retries,
		retry_passes=args.retry_passes,
	)

	failure_path = out_dir / "failed-downloads.txt"
	if failures:
		failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
		print(f"Wrote {len(failures)} failures to {failure_path}", file=sys.stderr)
		return 1

	failure_path.unlink(missing_ok=True)
	completion_path.write_text(f"{day.isoformat()}\n", encoding="utf-8")
	print(
		f"Done: downloaded={counts['downloaded']}, skipped={counts['skipped']}, failed=0",
		flush=True,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
