import stat

import pytest
from trino.auth import OAuth2Authentication
from trino.exceptions import TrinoAuthError

from data_engineering.opensky import trino_client


class FakeKeyring:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.tokens = {}

    def get_password(self, service, account):
        if self.fail:
            raise RuntimeError("keyring unavailable")
        return self.tokens.get((service, account))

    def set_password(self, service, account, token):
        if self.fail:
            raise RuntimeError("keyring unavailable")
        self.tokens[(service, account)] = token


def test_wait_for_external_authentication_waits_for_confirmation(
    monkeypatch,
    capsys,
):
    prompts = []
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: prompts.append(prompt) or "",
    )

    trino_client.wait_for_external_authentication("https://example.test/login")

    assert "https://example.test/login" in capsys.readouterr().out
    assert prompts == [
        "After authentication succeeds in the browser, press Enter here: "
    ]


def test_wait_for_external_authentication_explains_noninteractive_failure(
    monkeypatch,
):
    def raise_eof(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    with pytest.raises(TrinoAuthError, match="interactive terminal"):
        trino_client.wait_for_external_authentication(
            "https://example.test/login"
        )


def test_persistent_token_cache_uses_system_keyring(tmp_path):
    keyring = FakeKeyring()
    cache_path = tmp_path / "oauth.json"
    cache = trino_client.PersistentOAuthTokenCache(
        cache_path,
        keyring_module=keyring,
    )

    cache.store_token_to_cache("host@example", "secret-token")

    assert cache.get_token_from_cache("host@example") == "secret-token"
    assert keyring.tokens == {
        (
            trino_client.TOKEN_CACHE_SERVICE,
            "host@example",
        ): "secret-token"
    }
    assert not cache_path.exists()


@pytest.mark.parametrize("keyring", [None, FakeKeyring(fail=True)])
def test_persistent_token_cache_falls_back_to_private_file(
    tmp_path,
    keyring,
):
    cache_path = tmp_path / "state" / "oauth.json"
    cache = trino_client.PersistentOAuthTokenCache(
        cache_path,
        keyring_module=keyring,
        load_system_keyring=False,
    )

    cache.store_token_to_cache("host@example", "secret-token")
    fresh_cache = trino_client.PersistentOAuthTokenCache(
        cache_path,
        keyring_module=None,
        load_system_keyring=False,
    )

    assert fresh_cache.get_token_from_cache("host@example") == "secret-token"
    assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600


def test_default_token_cache_path_honors_environment(monkeypatch, tmp_path):
    configured_path = tmp_path / "custom-token.json"
    monkeypatch.setenv(
        trino_client.TOKEN_CACHE_PATH_ENV,
        str(configured_path),
    )

    assert trino_client.default_token_cache_path() == configured_path


def test_create_connection_configures_external_oauth(monkeypatch):
    captured = {}
    expected_connection = object()

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return expected_connection

    monkeypatch.setattr(trino_client, "connect", fake_connect)

    result = trino_client.create_connection("MixedCaseUser")

    assert result is expected_connection
    assert captured["user"] == "mixedcaseuser"
    assert isinstance(captured["auth"], OAuth2Authentication)
    assert (
        captured["auth"]._redirect_auth_url
        is trino_client.wait_for_external_authentication
    )
    assert isinstance(
        captured["auth"]._bearer._token_cache,
        trino_client.PersistentOAuthTokenCache,
    )
