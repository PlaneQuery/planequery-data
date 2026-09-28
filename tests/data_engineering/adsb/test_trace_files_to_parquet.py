from datetime import datetime

import pyarrow.parquet as pq

from data_engineering.adsb import trace_files_to_parquet as converter
from data_engineering.adsb.parquet_schema import COLUMNS, PARQUET_SCHEMA


def test_adsblol_writes_canonical_bucketed_day(monkeypatch, tmp_path):
    input_day = tmp_path / "extract/year=2024/month=06/day=01"
    input_day.mkdir(parents=True)
    trace_file = input_day / "trace_full_010000.json"
    trace_file.touch()

    columns = {column: [None] for column in COLUMNS}
    columns["time"] = [datetime(2024, 6, 1)]
    columns["icao"] = ["010000"]
    columns["lat"] = [40.0]
    columns["lon"] = [-75.0]
    monkeypatch.setattr(converter, "_trace_files", lambda folder: [trace_file])
    monkeypatch.setattr(
        converter,
        "_processed_trace_files",
        lambda files, max_workers, submit_chunk_size: iter([(trace_file, columns)]),
    )

    output_root = tmp_path / "canonical/source=adsblol/version=6"
    row_count = converter.process_trace_files(
        input_day,
        output_root=output_root,
        max_workers=1,
    )

    output_day = output_root / "year=2024/month=06/day=01"
    output_file = output_day / "icao_bucket=0/data.parquet"
    assert row_count == 1
    assert (output_day / "_SUCCESS").is_file()
    assert pq.read_schema(output_file).equals(PARQUET_SCHEMA)
    assert pq.read_table(output_file).column("icao").to_pylist() == ["010000"]
