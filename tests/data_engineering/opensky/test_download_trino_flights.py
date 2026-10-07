from datetime import date, datetime, timezone

import pyarrow.parquet as pq
import pytest

from data_engineering.opensky import download_trino_flights as downloader


class FakeCursor:
    description = [("icao24",), ("day",)]

    def __init__(self, batches=None, error=None):
        self.batches = iter(batches or [])
        self.error = error
        self.execute_args = None
        self.closed = False

    def execute(self, query, params):
        self.execute_args = (query, params)

    def fetchmany(self, batch_size):
        if self.error is not None:
            raise self.error
        return next(self.batches, [])

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, cursors=None):
        self.cursors = iter(cursors or [])
        self.closed = False

    def cursor(self):
        return next(self.cursors)

    def close(self):
        self.closed = True


def test_download_day_writes_partitioned_parquet(tmp_path):
    cursors = [
        FakeCursor(batches=[[("abc123", 1)]]),
        FakeCursor(batches=[[("def456", 1)]]),
    ]
    connection = FakeConnection(cursors)
    target_date = date(2026, 9, 26)

    output_path = downloader.download_day(
        target_date,
        connection=connection,
        output_dir=tmp_path,
        batch_size=10,
        query_window_minutes=12 * 60,
    )

    assert output_path == (
        tmp_path
        / "year=2026"
        / "month=09"
        / "day=26"
        / "data.parquet"
    )
    assert pq.read_table(output_path).column("icao24").to_pylist() == [
        "abc123",
        "def456",
    ]
    start = int(datetime(2026, 9, 26, tzinfo=timezone.utc).timestamp())
    first_query, first_params = cursors[0].execute_args
    assert "SELECT *" not in first_query
    assert "track," not in first_query.lower()
    assert "firstseen" in first_query
    assert "estdepartureairport" in first_query
    assert "otherarrivalairportcandidates" in first_query
    assert "serials" not in first_query
    assert "takeofflandinginfo" not in first_query
    assert "day" in first_query
    assert "FROM flights_data4" in first_query
    assert "WHERE day = ?" in first_query
    assert "AND lastseen >= ?" in first_query
    assert "AND lastseen < ?" in first_query
    assert first_params == [start, start, start + 12 * 60 * 60]
    assert cursors[1].execute_args[1] == [
        start,
        start + 12 * 60 * 60,
        start + 24 * 60 * 60,
    ]
    assert all(cursor.closed for cursor in cursors)
    assert not connection.closed


def test_download_day_skips_existing_output(tmp_path):
    target_date = date(2026, 9, 26)
    output_path = downloader.flights_output_path(target_date, tmp_path)
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"existing")

    result = downloader.download_day(
        target_date,
        connection=FakeConnection(),
        output_dir=tmp_path,
    )

    assert result == output_path
    assert output_path.read_bytes() == b"existing"


def test_download_day_overwrites_only_after_success(tmp_path):
    target_date = date(2026, 9, 26)
    output_path = downloader.flights_output_path(target_date, tmp_path)
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"existing")
    cursor = FakeCursor(batches=[[('abc123', 1)]])

    result = downloader.download_day(
        target_date,
        connection=FakeConnection([cursor]),
        output_dir=tmp_path,
        query_window_minutes=24 * 60,
        overwrite=True,
    )

    assert result == output_path
    assert pq.read_table(result).column("icao24").to_pylist() == ["abc123"]


def test_failed_overwrite_preserves_completed_output(tmp_path):
    target_date = date(2026, 9, 26)
    output_path = downloader.flights_output_path(target_date, tmp_path)
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"existing")

    with pytest.raises(RuntimeError, match="query failed"):
        downloader.download_day(
            target_date,
            connection=FakeConnection(
                [FakeCursor(error=RuntimeError("query failed"))]
            ),
            output_dir=tmp_path,
            query_window_minutes=24 * 60,
            overwrite=True,
        )

    assert output_path.read_bytes() == b"existing"
    assert not output_path.with_suffix(".parquet.tmp").exists()


def test_download_day_removes_temporary_file_after_failure(tmp_path):
    cursor = FakeCursor(error=RuntimeError("query failed"))
    target_date = date(2026, 9, 26)
    output_path = downloader.flights_output_path(target_date, tmp_path)

    with pytest.raises(RuntimeError, match="query failed"):
        downloader.download_day(
            target_date,
            connection=FakeConnection([cursor]),
            output_dir=tmp_path,
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".parquet.tmp").exists()
    assert cursor.closed


def test_download_day_closes_owned_connection(monkeypatch, tmp_path):
    cursor = FakeCursor()
    connection = FakeConnection([cursor])
    monkeypatch.setattr(downloader, "create_connection", lambda user: connection)

    downloader.download_day(
        date(2026, 9, 26),
        user="test-user",
        output_dir=tmp_path,
        query_window_minutes=24 * 60,
    )

    assert connection.closed


def test_main_passes_cli_arguments(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        downloader,
        "download_day",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    downloader.main(
        [
            "2026-09-26",
            "--user",
            "test-user",
            "--output-dir",
            str(tmp_path),
            "--batch-size",
            "42",
            "--query-window-minutes",
            "30",
            "--overwrite",
        ]
    )

    assert calls == [
        (
            (date(2026, 9, 26),),
            {
                "user": "test-user",
                "output_dir": tmp_path,
                "batch_size": 42,
                "query_window_minutes": 30,
                "overwrite": True,
            },
        )
    ]


def test_main_downloads_date_range_with_exclusive_end(monkeypatch):
    calls = []
    monkeypatch.setattr(
        downloader,
        "download_day",
        lambda target_date, **kwargs: calls.append(target_date),
    )

    downloader.main(["2026-09-26", "2026-09-29"])

    assert calls == [
        date(2026, 9, 26),
        date(2026, 9, 27),
        date(2026, 9, 28),
    ]


@pytest.mark.parametrize("end_date", ["2026-09-26", "2026-09-25"])
def test_parse_args_rejects_non_increasing_date_range(end_date, capsys):
    with pytest.raises(SystemExit):
        downloader.parse_args(["2026-09-26", end_date])

    assert "END_DATE must be after START_DATE" in capsys.readouterr().err
