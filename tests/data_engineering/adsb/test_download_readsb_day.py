from data_engineering.adsbx import download_readsb_day


def test_download_tasks_retries_only_failures(monkeypatch, tmp_path):
    attempts = {}

    def fake_download_one(url, target, timeout, retries):
        attempts[url] = attempts.get(url, 0) + 1
        if url == "retry" and attempts[url] == 1:
            return "failed", "retry :: temporary failure"
        return "downloaded", url

    monkeypatch.setattr(download_readsb_day, "download_one", fake_download_one)
    monkeypatch.setattr(download_readsb_day.time, "sleep", lambda _seconds: None)
    tasks = [
        ("ok", tmp_path / "ok.json.gz"),
        ("retry", tmp_path / "retry.json.gz"),
    ]

    counts, failures = download_readsb_day.download_tasks(
        tasks,
        workers=2,
        timeout=1,
        retries=0,
        retry_passes=3,
    )

    assert counts == {"downloaded": 2, "skipped": 0, "failed": 0}
    assert failures == []
    assert attempts == {"ok": 1, "retry": 2}


def test_incomplete_day_runs_downloader_again(monkeypatch, tmp_path):
    from data_engineering.adsbx import readsb

    day = readsb.dt.date(2025, 7, 1)
    input_root = tmp_path / "readsb-hist"
    day_dir = input_root / "2025" / "07" / "01"
    day_dir.mkdir(parents=True)
    (day_dir / "000000Z.json.gz").write_bytes(b"partial")
    calls = []

    def fake_run(cmd, check):
        calls.append((cmd, check))
        download_readsb_day.download_complete_path(day_dir).write_text(
            "2025-07-01\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(readsb, "DEFAULT_INPUT_ROOT", input_root)
    monkeypatch.setattr(readsb.subprocess, "run", fake_run)

    assert readsb._ensure_day_downloaded(day) == day_dir
    assert len(calls) == 1
    assert calls[0][1] is True


def test_complete_day_does_not_run_downloader(monkeypatch, tmp_path):
    from data_engineering.adsbx import readsb

    day = readsb.dt.date(2025, 7, 1)
    input_root = tmp_path / "readsb-hist"
    day_dir = input_root / "2025" / "07" / "01"
    day_dir.mkdir(parents=True)
    (day_dir / "000000Z.json.gz").write_bytes(b"snapshot")
    download_readsb_day.download_complete_path(day_dir).write_text(
        "2025-07-01\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(readsb, "DEFAULT_INPUT_ROOT", input_root)
    monkeypatch.setattr(
        readsb.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected run")),
    )

    assert readsb._ensure_day_downloaded(day) == day_dir
