import io
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from types import SimpleNamespace
from pathlib import Path

import pytest

from data_engineering.adsb import download_adsb_to_parquet as download


class FakeResponse:
    def __init__(self, body: bytes, content_length: int | None):
        self._stream = io.BytesIO(body)
        self.status = 200
        self.reason = "OK"
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def test_download_asset_rejects_truncated_response(monkeypatch, tmp_path):
    destination = tmp_path / "release.tar.aa"
    monkeypatch.setattr(
        download.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(b"partial", content_length=100),
    )
    monkeypatch.setattr(download.time, "sleep", lambda _: None)

    assert not download.download_asset("https://example.test/release", destination)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_asset_atomically_replaces_wrong_sized_file(monkeypatch, tmp_path):
    destination = tmp_path / "release.tar.aa"
    destination.write_bytes(b"old")
    body = b"complete release asset"
    monkeypatch.setattr(
        download.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(body, content_length=len(body)),
    )

    assert download.download_asset("https://example.test/release", destination)
    assert destination.read_bytes() == body
    assert not list(tmp_path.glob("*.part"))


def test_download_asset_requires_content_length(monkeypatch, tmp_path):
    destination = tmp_path / "release.tar.aa"
    monkeypatch.setattr(
        download.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(b"unverifiable", content_length=None),
    )
    monkeypatch.setattr(download.time, "sleep", lambda _: None)

    assert not download.download_asset("https://example.test/release", destination)
    assert not destination.exists()


def test_release_download_slots_limit_concurrency(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "LOCKS_DIR", tmp_path / "locks")
    (tmp_path / "locks").mkdir()
    state_lock = threading.Lock()
    first_two_entered = threading.Event()
    release_workers = threading.Event()
    active = 0
    maximum_active = 0
    completed = 0

    def worker():
        nonlocal active, maximum_active, completed
        with download._release_download_slot():
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
                if active == download.DOWNLOAD_CONCURRENCY:
                    first_two_entered.set()
            release_workers.wait(timeout=2)
            with state_lock:
                active -= 1
                completed += 1

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()

    assert first_two_entered.wait(timeout=2)
    time.sleep(0.05)
    with state_lock:
        assert active == 2
        assert maximum_active == 2

    release_workers.set()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert completed == 3
    assert maximum_active == 2


def _write_split_tar(tmp_path: Path) -> tuple[list[Path], int]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "first.txt").write_text("first", encoding="utf-8")
    (source_dir / "second.bin").write_bytes(b"x" * (1024 * 1024))

    archive = tmp_path / "release.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(source_dir / "first.txt", arcname="release/first.txt")
        tar.add(source_dir / "second.bin", arcname="release/second.bin")

    with tarfile.open(archive) as tar:
        second_data_offset = tar.getmember("release/second.bin").offset_data

    contents = archive.read_bytes()
    split_at = len(contents) // 2
    first_part = tmp_path / "release.tar.aa"
    second_part = tmp_path / "release.tar.ab"
    first_part.write_bytes(contents[:split_at])
    second_part.write_bytes(contents[split_at:])
    return [first_part, second_part], second_data_offset


def test_extract_split_archive_is_atomic_and_marked(tmp_path):
    parts, _ = _write_split_tar(tmp_path)
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    (extract_dir / "partial.txt").write_text("stale", encoding="utf-8")

    assert download.extract_split_archive(parts, extract_dir)
    assert (extract_dir / "first.txt").read_text(encoding="utf-8") == "first"
    assert (extract_dir / "second.bin").stat().st_size == 1024 * 1024
    assert (extract_dir / download.EXTRACTION_COMPLETE_MARKER).is_file()
    assert not (extract_dir / "partial.txt").exists()


def test_extract_split_archive_rejects_partial_output(tmp_path):
    parts, second_data_offset = _write_split_tar(tmp_path)
    truncated_part = tmp_path / "truncated.tar.aa"
    complete_archive = b"".join(path.read_bytes() for path in parts)
    truncated_part.write_bytes(complete_archive[: second_data_offset + 100])
    extract_dir = tmp_path / "extract"

    assert not download.extract_split_archive([truncated_part], extract_dir)
    assert not extract_dir.exists()
    assert not list(tmp_path.glob(".extract.extract-*"))


def test_extract_prefers_split_parts_over_standalone_tar(tmp_path, capsys):
    parts, _ = _write_split_tar(tmp_path)
    archive = tmp_path / "release.tar"
    # The standalone version must not be read, even if it is invalid.
    archive.write_bytes(b"invalid standalone archive" * 100_000)
    extract_dir = tmp_path / "extract"

    assert download.extract_split_archive([archive, *reversed(parts)], extract_dir)
    assert (extract_dir / "first.txt").read_text() == "first"
    assert (extract_dir / "second.bin").read_bytes() == b"x" * (1024 * 1024)
    output = capsys.readouterr().out
    assert "Using split archive parts for release.tar" in output
    assert "Ignoring standalone tar because split parts are available:" in output
    marker = (extract_dir / download.EXTRACTION_COMPLETE_MARKER).read_text()
    assert '"name": "release.tar"' not in marker


def test_extract_uses_standalone_tar_without_split_parts(tmp_path, capsys):
    _write_split_tar(tmp_path)
    extract_dir = tmp_path / "extract"

    assert download.extract_split_archive([tmp_path / "release.tar"], extract_dir)
    assert (extract_dir / "first.txt").read_text() == "first"
    assert (extract_dir / "second.bin").read_bytes() == b"x" * (1024 * 1024)
    assert "Using standalone tar because no split parts are available:" in capsys.readouterr().out


def test_upload_includes_success_marker(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    parquet_day = output_dir / "data/processed/adsb/parquet_output/day=01"
    for bucket in range(8):
        path = parquet_day / f"icao_bucket={bucket}/data.parquet"
        path.parent.mkdir(parents=True)
        path.touch()
    (parquet_day / "_SUCCESS").touch()

    uploads = []
    fake_s3 = SimpleNamespace(
        upload_file=lambda path, bucket, key: uploads.append((path, bucket, key))
    )
    fake_boto3 = SimpleNamespace(client=lambda service: fake_s3)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setattr(download, "OUTPUT_DIR", output_dir)

    download.upload_to_s3(parquet_day)

    assert len(uploads) == 9
    assert uploads[-1][2].endswith("/day=01/_SUCCESS")


def test_concurrent_day_conversion_runs_once(monkeypatch, tmp_path):
    partition_dir = tmp_path / "parquet"
    monkeypatch.setattr(download, "LOCKS_DIR", tmp_path / "locks")
    monkeypatch.setattr(download, "_parquet_partition_dir", lambda day: partition_dir)
    conversion_started = threading.Event()
    finish_conversion = threading.Event()
    second_checked_partition = threading.Event()
    calls = []
    has_data = download._parquet_partition_has_data

    def checked_partition(path):
        if conversion_started.is_set() and not finish_conversion.is_set():
            second_checked_partition.set()
        return has_data(path)

    def convert(version_date, keep_folders):
        calls.append(version_date)
        conversion_started.set()
        assert finish_conversion.wait(timeout=5)
        partition_dir.mkdir(exist_ok=True)
        (partition_dir / download.SUCCESS_MARKER).touch()
        return 1

    monkeypatch.setattr(download, "_parquet_partition_has_data", checked_partition)
    monkeypatch.setattr(download, "process_version_date", convert)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(download.create_parquet_for_day, date(2024, 3, 22))
        try:
            assert conversion_started.wait(timeout=5)
            second = executor.submit(download.create_parquet_for_day, date(2024, 3, 22))
            assert second_checked_partition.wait(timeout=5)
        finally:
            finish_conversion.set()
        assert first.result(timeout=5) == partition_dir
        assert second.result(timeout=5) == partition_dir

    assert calls == ["v2024.03.22"]


@pytest.mark.parametrize("failure", ["missing_release", "download", "extraction"])
def test_release_failure_identifies_stage(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(download, "RELEASES_DIR", tmp_path / "releases")
    monkeypatch.setattr(download, "EXTRACT_DIR", tmp_path / "extract")
    monkeypatch.setattr(download, "LOCKS_DIR", tmp_path / "locks")
    url = "https://example.test/v2024.03.22.tar.aa"
    monkeypatch.setattr(
        download, "fetch_release_urls",
        lambda version: [] if failure == "missing_release" else [url],
    )
    monkeypatch.setattr(download, "download_asset", lambda *args: failure != "download")
    monkeypatch.setattr(download, "extract_split_archive", lambda *args: False)

    messages = {
        "missing_release": "No adsblol production releases found for v2024.03.22",
        "download": "Failed to download adsblol archive for v2024.03.22",
        "extraction": "Failed to extract adsblol archives for v2024.03.22",
    }
    with pytest.raises((FileNotFoundError, RuntimeError), match=messages[failure]) as exc:
        download.download_and_extract("v2024.03.22")
    if failure == "download":
        assert url in str(exc.value)
