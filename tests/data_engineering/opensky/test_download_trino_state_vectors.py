from datetime import date, datetime, timezone

import pyarrow.parquet as pq
import pyarrow as pa
import pytest

from data_engineering.opensky import download_trino_state_vectors as downloader


class FakeCursor:
    description = [("icao24", "varchar"), ("hour", "bigint")]

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


def test_rows_to_arrow_preserves_columns():
    table = downloader.rows_to_arrow(
        [("abc123", 1), ("def456", 2)],
        ["icao24", "hour"],
    )

    assert table.to_pydict() == {
        "icao24": ["abc123", "def456"],
        "hour": [1, 2],
    }


def test_download_hour_writes_partitioned_parquet(tmp_path):
    cursors = [
        FakeCursor(batches=[[("abc123", 1)]]),
        FakeCursor(batches=[[("def456", 1)]]),
    ]
    start = datetime(2024, 6, 1, 3, tzinfo=timezone.utc)

    output_path = downloader.download_hour(
        FakeConnection(cursors),
        start,
        output_dir=tmp_path,
        batch_size=10,
        query_window_minutes=30,
    )

    assert output_path == (
        tmp_path
        / "year=2024"
        / "month=06"
        / "day=01"
        / "hour=03"
        / "data.parquet"
    )
    assert pq.read_table(output_path).column("icao24").to_pylist() == [
        "abc123",
        "def456",
    ]
    assert cursors[0].execute_args[1] == [
        int(start.timestamp()),
        int(start.timestamp()),
        int(start.timestamp()) + 30 * 60,
    ]
    assert cursors[1].execute_args[1] == [
        int(start.timestamp()),
        int(start.timestamp()) + 30 * 60,
        int(start.timestamp()) + 60 * 60,
    ]
    assert all(cursor.closed for cursor in cursors)
    assert not output_path.with_suffix(".parquet.tmp").exists()


@pytest.mark.parametrize("nulls_first", [True, False])
def test_download_hour_preserves_nullable_types_across_windows(tmp_path, nulls_first):
    populated = (1234.5, "ABC123", True, [42])
    missing = (None, None, None, [])
    rows = [missing, populated] if nulls_first else [populated, missing]
    cursors = [FakeCursor(batches=[[row]]) for row in rows]
    for cursor in cursors:
        cursor.description = [
            ("baroaltitude", "double"),
            ("callsign", "varchar(8)"),
            ("onground", "boolean"),
            ("serials", "array(integer)"),
        ]

    output_path = downloader.download_hour(
        FakeConnection(cursors),
        datetime(2026, 2, 28, 7, tzinfo=timezone.utc),
        output_dir=tmp_path,
        query_window_minutes=30,
    )

    table = pq.ParquetFile(output_path).read()
    assert table.schema == pa.schema([
        ("baroaltitude", pa.float64()),
        ("callsign", pa.string()),
        ("onground", pa.bool_()),
        ("serials", pa.list_(pa.int64())),
    ])
    assert table.to_pydict() == dict(zip(table.column_names, map(list, zip(*rows))))


def test_download_empty_hour_preserves_declared_types(tmp_path):
    output_path = downloader.download_hour(
        FakeConnection([FakeCursor()]),
        datetime(2026, 2, 28, tzinfo=timezone.utc),
        output_dir=tmp_path,
        query_window_minutes=60,
    )

    table = pq.ParquetFile(output_path).read()
    assert table.num_rows == 0
    assert table.schema == pa.schema([("icao24", pa.string()), ("hour", pa.int64())])


def test_download_hour_skips_existing_output(tmp_path):
    output_path = downloader.state_vectors_output_path(
        datetime(2024, 6, 1, tzinfo=timezone.utc),
        tmp_path,
    )
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"existing")

    result = downloader.download_hour(
        FakeConnection(),
        datetime(2024, 6, 1, tzinfo=timezone.utc),
        output_dir=tmp_path,
    )

    assert result == output_path
    assert output_path.read_bytes() == b"existing"


def test_download_hour_removes_temporary_file_after_failure(tmp_path):
    cursor = FakeCursor(error=RuntimeError("query failed"))
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    output_path = downloader.state_vectors_output_path(start, tmp_path)

    with pytest.raises(RuntimeError, match="query failed"):
        downloader.download_hour(
            FakeConnection([cursor]),
            start,
            output_dir=tmp_path,
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".parquet.tmp").exists()
    assert cursor.closed


def test_download_day_uses_each_utc_hour_and_closes_owned_connection(
    monkeypatch,
    tmp_path,
):
    connection = FakeConnection()
    starts = []
    monkeypatch.setattr(downloader, "create_connection", lambda user: connection)
    monkeypatch.setattr(
        downloader,
        "download_hour",
        lambda connection, start, **kwargs: starts.append(start) or tmp_path,
    )

    outputs = downloader.download_day(
        date(2024, 6, 1),
        user="test-user",
        output_dir=tmp_path,
    )

    assert len(outputs) == 24
    assert starts[0] == datetime(2024, 6, 1, tzinfo=timezone.utc)
    assert starts[-1] == datetime(2024, 6, 1, 23, tzinfo=timezone.utc)
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
            "2024-06-01",
            "--user",
            "test-user",
            "--output-dir",
            str(tmp_path),
            "--batch-size",
            "42",
            "--query-window-minutes",
            "5",
        ]
    )

    assert calls == [
        (
            (date(2024, 6, 1),),
            {
                "user": "test-user",
                "output_dir": tmp_path,
                "batch_size": 42,
                "query_window_minutes": 5,
            },
        )
    ]
