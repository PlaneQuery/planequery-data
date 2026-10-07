import pytest
import requests
from trino.exceptions import TrinoAuthError

from data_engineering.opensky import trino_client


class FakeResponse:
    def __init__(self, payload=None, *, status_code=200, json_error=None):
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


def test_token_provider_uses_password_grant_and_caches_token():
    session = FakeSession(
        [FakeResponse({"access_token": "token-1", "expires_in": 600})]
    )
    now = [100.0]
    provider = trino_client.OpenSkyTokenProvider(
        "MixedCaseUser",
        "secret-password",
        session=session,
        clock=lambda: now[0],
    )

    assert provider.get_token() == "token-1"
    now[0] = 500.0
    assert provider.get_token() == "token-1"

    assert session.calls == [
        (
            trino_client.OPENSKY_TOKEN_URL,
            {
                "data": {
                    "client_id": trino_client.OPENSKY_TRINO_CLIENT_ID,
                    "grant_type": "password",
                    "username": "mixedcaseuser",
                    "password": "secret-password",
                },
                "timeout": trino_client.TOKEN_REQUEST_TIMEOUT_SECONDS,
            },
        )
    ]


def test_token_provider_refreshes_before_expiration():
    session = FakeSession(
        [
            FakeResponse({"access_token": "token-1", "expires_in": 600}),
            FakeResponse({"access_token": "token-2", "expires_in": 600}),
        ]
    )
    now = [100.0]
    provider = trino_client.OpenSkyTokenProvider(
        "test-user",
        "secret-password",
        session=session,
        clock=lambda: now[0],
    )

    assert provider.get_token() == "token-1"
    now[0] = 640.0
    assert provider.get_token() == "token-2"
    assert len(session.calls) == 2


def test_refreshing_auth_uses_current_token_for_every_request():
    class FakeProvider:
        def __init__(self):
            self.tokens = iter(["token-1", "token-2"])

        def get_token(self):
            return next(self.tokens)

    session = requests.Session()
    authentication = trino_client.RefreshingJWTAuthentication(FakeProvider())
    authentication.set_http_session(session)
    first = requests.Request("GET", "https://example.test/first").prepare()
    second = requests.Request("GET", "https://example.test/second").prepare()

    session.auth(first)
    session.auth(second)

    assert first.headers["Authorization"] == "Bearer token-1"
    assert second.headers["Authorization"] == "Bearer token-2"


@pytest.mark.parametrize(
    ("username", "password", "missing"),
    [
        (None, "secret-password", "OPENSKY_USERNAME"),
        ("test-user", None, "OPENSKY_PASSWORD"),
        (None, None, "OPENSKY_USERNAME and OPENSKY_PASSWORD"),
    ],
)
def test_credentials_explain_missing_environment(
    monkeypatch,
    username,
    password,
    missing,
):
    monkeypatch.delenv(trino_client.OPENSKY_USERNAME_ENV, raising=False)
    monkeypatch.delenv(trino_client.OPENSKY_PASSWORD_ENV, raising=False)
    if username is not None:
        monkeypatch.setenv(trino_client.OPENSKY_USERNAME_ENV, username)
    if password is not None:
        monkeypatch.setenv(trino_client.OPENSKY_PASSWORD_ENV, password)

    with pytest.raises(TrinoAuthError, match=missing):
        trino_client.opensky_credentials_from_environment()


def test_token_provider_explains_rejected_credentials():
    provider = trino_client.OpenSkyTokenProvider(
        "test-user",
        "wrong-password",
        session=FakeSession([FakeResponse(status_code=401)]),
    )

    with pytest.raises(TrinoAuthError, match="rejected"):
        provider.get_token()


def test_create_connection_configures_refreshing_auth(monkeypatch):
    captured = {}
    expected_connection = object()

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return expected_connection

    monkeypatch.setenv(trino_client.OPENSKY_USERNAME_ENV, "MixedCaseUser")
    monkeypatch.setenv(
        trino_client.OPENSKY_PASSWORD_ENV,
        "secret-password",
    )
    monkeypatch.setattr(trino_client, "connect", fake_connect)

    result = trino_client.create_connection("mixedcaseuser")

    assert result is expected_connection
    assert captured["user"] == "mixedcaseuser"
    assert isinstance(
        captured["auth"],
        trino_client.RefreshingJWTAuthentication,
    )
    assert captured["auth"].token_provider.username == "mixedcaseuser"


def test_create_connection_rejects_mismatched_user(monkeypatch):
    monkeypatch.setenv(trino_client.OPENSKY_USERNAME_ENV, "credential-user")
    monkeypatch.setenv(
        trino_client.OPENSKY_PASSWORD_ENV,
        "secret-password",
    )

    with pytest.raises(TrinoAuthError, match="does not match"):
        trino_client.create_connection("different-user")
