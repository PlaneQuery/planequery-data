import gzip

from data_engineering.adsb.process_trace_file import process_file


def test_process_file_skips_corrupt_gzip(tmp_path):
    trace_file = tmp_path / "trace_full_a00001.json"
    trace_file.write_bytes(b"\x1f\x8bcorrupt gzip bytes")

    assert process_file(trace_file) is None


def test_process_file_skips_invalid_json(tmp_path):
    trace_file = tmp_path / "trace_full_a00001.json"
    trace_file.write_bytes(gzip.compress(b"{not json"))

    assert process_file(trace_file) is None
