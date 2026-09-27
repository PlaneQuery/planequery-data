from types import SimpleNamespace

from data_engineering.adsblol import check_for_new_release as releases


def test_cache_freshness_uses_modification_time(monkeypatch, tmp_path):
    cache_path = tmp_path / "ALL_RELEASES_2026.txt"
    cache_path.write_text("release", encoding="utf-8")
    monkeypatch.setattr(releases.time, "time", lambda: 1_000)
    monkeypatch.setattr(
        releases.Path,
        "stat",
        lambda _: SimpleNamespace(st_mtime=999, st_birthtime=0, st_ctime=0),
    )

    assert releases._cache_is_fresh(cache_path)


def test_prepared_release_catalog_is_reused_without_network(monkeypatch, tmp_path):
    monkeypatch.setattr(releases, "ADSBLOL_DIR", tmp_path)
    monkeypatch.setattr(releases, "_RUN_RELEASES_CACHE", {})
    monkeypatch.delenv(releases.PREPARED_RELEASE_YEARS_ENV, raising=False)
    requests = []
    content = (
        "https://github.com/adsblol/globe_history_2026/releases/download/"
        "v2026.07.26-planes-readsb-prod-0/"
        "v2026.07.26-planes-readsb-prod-0.tar.aa\n"
    )

    def request_text(url):
        requests.append(url)
        return content

    monkeypatch.setattr(releases, "_request_text", request_text)
    releases.prepare_releases_for_run([2026])
    assert len(requests) == 2  # ALL_RELEASES.txt and releases.atom, once each.

    # Simulate a spawned worker: process memory is empty, but the inherited
    # run marker points it at the parent's prepared on-disk snapshot.
    releases._RUN_RELEASES_CACHE.clear()
    monkeypatch.setattr(
        releases,
        "_request_text",
        lambda _: (_ for _ in ()).throw(AssertionError("network must not be used")),
    )

    assert releases.load_releases_txt("2026") == content
