import pytest

from flights import workers


def test_worker_count_is_limited_by_ram(monkeypatch):
    monkeypatch.setattr(workers.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(workers, "total_memory_bytes", lambda: 16 * workers.GIB)

    assert workers.default_worker_count() == 3


def test_worker_count_is_limited_by_cpu(monkeypatch):
    monkeypatch.setattr(workers.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(workers, "total_memory_bytes", lambda: 128 * workers.GIB)

    assert workers.default_worker_count() == 4


def test_worker_count_falls_back_to_cpu_when_ram_is_unknown(monkeypatch):
    monkeypatch.setattr(workers.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(workers, "total_memory_bytes", lambda: None)

    assert workers.default_worker_count() == 6


def test_worker_count_is_always_at_least_one(monkeypatch):
    monkeypatch.setattr(workers.os, "cpu_count", lambda: None)
    monkeypatch.setattr(workers, "total_memory_bytes", lambda: workers.GIB)

    assert workers.default_worker_count() == 1


def test_worker_count_rejects_invalid_memory_estimate():
    with pytest.raises(ValueError, match="must be positive"):
        workers.default_worker_count(memory_per_worker_bytes=0)
