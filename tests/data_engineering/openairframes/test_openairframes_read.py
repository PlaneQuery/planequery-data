from datetime import date
from io import BytesIO
import json
import os
from pathlib import Path
import urllib.error

import pytest

os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[3]))

from data_engineering.openairframes import read


@pytest.fixture(autouse=True)
def clear_snapshot_caches():
    cached_functions = (
        read._remote_snapshot_assets,
        read._resolve_default_snapshot,
    )
    for function in cached_functions:
        function.cache_clear()
    yield
    for function in cached_functions:
        function.cache_clear()


def test_remote_assets_use_each_dataset_date(monkeypatch):
    releases = [
        {
            "tag_name": "openairframes-2026-10-01-main",
            "draft": False,
            "assets": [
                {
                    "name": "openairframes_adsb_2024-01-01_2026-09-30.csv.gz",
                    "browser_download_url": "https://example.test/too-new-adsb",
                },
                {
                    "name": "openairframes_faa_2023-08-16_2026-10-01.csv",
                    "browser_download_url": "https://example.test/too-new-faa",
                },
            ],
        },
        {
            "tag_name": "openairframes-2026-09-29-main",
            "draft": False,
            "assets": [
                {
                    "name": "openairframes_faa_2023-08-16_2026-09-29.csv",
                    "browser_download_url": "https://example.test/faa",
                },
                {
                    "name": "openairframes_adsb_2024-01-01_2026-09-28.csv.gz",
                    "browser_download_url": "https://example.test/adsb",
                },
            ],
        },
    ]

    def fake_urlopen(request, timeout):
        assert request.full_url == read.OPENAIRFRAMES_RELEASES_API_URL
        assert timeout == 30
        return BytesIO(json.dumps(releases).encode())

    monkeypatch.setattr(read.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(read, "OPENAIRFRAMES_RELEASE_DATE", date(2026, 9, 29))

    faa_asset = read._remote_snapshot_assets("faa")[0]
    adsb_asset = read._remote_snapshot_assets("adsb")[0]

    assert faa_asset.release_date == date(2026, 9, 29)
    assert faa_asset.snapshot_date == date(2026, 9, 29)
    assert adsb_asset.release_date == date(2026, 9, 29)
    assert adsb_asset.snapshot_date == date(2026, 9, 28)


def test_asset_discovery_skips_an_incomplete_latest_release(monkeypatch):
    releases = [
        {
            "tag_name": "openairframes-2026-09-30-main",
            "draft": False,
            "assets": [
                {
                    "name": "openairframes_community_2026-02-18_2026-09-30.csv",
                    "browser_download_url": "https://example.test/community",
                }
            ],
        },
        {
            "tag_name": "openairframes-2026-09-29-main",
            "draft": False,
            "assets": [
                {
                    "name": "openairframes_faa_2023-08-16_2026-09-29.csv",
                    "browser_download_url": "https://example.test/faa",
                }
            ],
        },
    ]

    monkeypatch.setattr(read, "OPENAIRFRAMES_RELEASE_DATE", date(2026, 9, 30))
    monkeypatch.setattr(
        read.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: BytesIO(json.dumps(releases).encode()),
    )

    asset = read._remote_snapshot_assets("faa")[0]

    assert asset.release_date == date(2026, 9, 29)
    assert asset.name == "openairframes_faa_2023-08-16_2026-09-29.csv"


def test_missing_newest_asset_download_falls_back_to_earlier_date(
    monkeypatch, tmp_path
):
    newest = read._SnapshotAsset(
        release_date=date(2026, 9, 30),
        snapshot_date=date(2026, 9, 30),
        name="openairframes_faa_2023-08-16_2026-09-30.csv",
        download_url="https://example.test/newest",
    )
    earlier = read._SnapshotAsset(
        release_date=date(2026, 9, 29),
        snapshot_date=date(2026, 9, 29),
        name="openairframes_faa_2023-08-16_2026-09-29.csv",
        download_url="https://example.test/earlier",
    )
    attempts = []

    def fake_download(asset, destination):
        attempts.append(asset.name)
        if asset is newest:
            raise urllib.error.HTTPError(
                asset.download_url, 404, "Not Found", None, None
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("snapshot", encoding="utf-8")

    monkeypatch.setattr(read, "OPENAIRFRAMES_DIR", tmp_path)
    monkeypatch.setattr(
        read, "_remote_snapshot_assets", lambda dataset: [newest, earlier]
    )
    monkeypatch.setattr(read, "_download_snapshot", fake_download)

    resolved = read._resolve_default_snapshot("faa")

    assert resolved == tmp_path / earlier.name
    assert attempts == [newest.name, earlier.name]


def test_release_lookup_failure_uses_latest_nonempty_local_snapshot(
    monkeypatch, tmp_path
):
    older = tmp_path / "openairframes_faa_2023-08-16_2026-09-28.csv"
    newer = tmp_path / "openairframes_faa_2023-08-16_2026-09-29.csv"
    empty = tmp_path / "openairframes_faa_2023-08-16_2026-09-30.csv"
    older.write_text("older", encoding="utf-8")
    newer.write_text("newer", encoding="utf-8")
    empty.touch()

    def fail_lookup(dataset):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(read, "OPENAIRFRAMES_DIR", tmp_path)
    monkeypatch.setattr(read, "_remote_snapshot_assets", fail_lookup)

    with pytest.warns(UserWarning, match="using local snapshot"):
        resolved = read._resolve_default_snapshot("faa")

    assert resolved == newer


def test_existing_asset_from_latest_release_is_reused(monkeypatch, tmp_path):
    asset = read._SnapshotAsset(
        release_date=date(2026, 9, 29),
        snapshot_date=date(2026, 9, 29),
        name="openairframes_faa_2023-08-16_2026-09-29.csv",
        download_url="https://example.test/faa",
    )
    cached = tmp_path / asset.name
    cached.write_text("snapshot", encoding="utf-8")

    monkeypatch.setattr(read, "OPENAIRFRAMES_DIR", tmp_path)
    monkeypatch.setattr(read, "_remote_snapshot_assets", lambda dataset: [asset])
    monkeypatch.setattr(
        read,
        "_download_snapshot",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not download")),
    )

    assert read._resolve_default_snapshot("faa") == cached
