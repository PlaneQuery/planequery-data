"""Shared OpenSky Trino connection and Arrow helpers."""

from collections.abc import Callable
import os
import threading
import time
from typing import Any

import pyarrow as pa
import requests
from requests.auth import AuthBase
from trino.auth import Authentication
from trino.dbapi import Connection, connect
from trino.exceptions import TrinoAuthError


BATCH_SIZE = 100_000
OPENSKY_USERNAME_ENV = "OPENSKY_USERNAME"
OPENSKY_PASSWORD_ENV = "OPENSKY_PASSWORD"
DEFAULT_TRINO_USER = os.environ.get(
    OPENSKY_USERNAME_ENV,
    "",
)
OPENSKY_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/"
    "protocol/openid-connect/token"
)
OPENSKY_TRINO_CLIENT_ID = "trino-client"
TOKEN_REQUEST_TIMEOUT_SECONDS = 30
TOKEN_REFRESH_MARGIN_SECONDS = 60
DEFAULT_TOKEN_LIFETIME_SECONDS = 300


class OpenSkyTokenProvider:
    """Acquire and proactively refresh an OpenSky Trino bearer token."""

    def __init__(
        self,
        username: str,
        password: str,
        *,
        session: requests.Session | None = None,
        clock: Callable[[], float] = time.monotonic,
        refresh_margin_seconds: float = TOKEN_REFRESH_MARGIN_SECONDS,
    ) -> None:
        if not username:
            raise ValueError("username must not be empty")
        if not password:
            raise ValueError("password must not be empty")
        if refresh_margin_seconds < 0:
            raise ValueError("refresh_margin_seconds must not be negative")

        self.username = username.lower()
        self._password = password
        self._session = session or requests.Session()
        self._clock = clock
        self._refresh_margin_seconds = refresh_margin_seconds
        self._lock = threading.Lock()
        self._token: str | None = None
        self._refresh_at = 0.0

    def get_token(self) -> str:
        """Return a valid token, refreshing it before it expires."""
        with self._lock:
            now = self._clock()
            if self._token is not None and now < self._refresh_at:
                return self._token

            token, lifetime = self._request_token()
            # For unusually short-lived tokens, use a 10% margin instead of
            # refreshing on every request.
            margin = min(self._refresh_margin_seconds, lifetime / 10)
            self._token = token
            self._refresh_at = now + max(lifetime - margin, 1)
            return token

    def _request_token(self) -> tuple[str, float]:
        try:
            response = self._session.post(
                OPENSKY_TOKEN_URL,
                data={
                    "client_id": OPENSKY_TRINO_CLIENT_ID,
                    "grant_type": "password",
                    "username": self.username,
                    "password": self._password,
                },
                timeout=TOKEN_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            raise TrinoAuthError(
                "Could not reach the OpenSky authentication service"
            ) from error

        if response.status_code in (400, 401):
            raise TrinoAuthError(
                "OpenSky rejected OPENSKY_USERNAME/OPENSKY_PASSWORD. "
                "Verify that they are the website credentials for the same "
                "account that has Trino access."
            )

        try:
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise TrinoAuthError(
                "OpenSky returned an invalid authentication response"
            ) from error

        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise TrinoAuthError(
                "OpenSky authentication response did not contain an access token"
            )

        expires_in = payload.get(
            "expires_in",
            DEFAULT_TOKEN_LIFETIME_SECONDS,
        )
        try:
            lifetime = float(expires_in)
        except (TypeError, ValueError):
            lifetime = DEFAULT_TOKEN_LIFETIME_SECONDS
        if lifetime <= 0:
            lifetime = DEFAULT_TOKEN_LIFETIME_SECONDS
        return token, lifetime


class _RefreshingBearerAuth(AuthBase):
    """Inject the provider's current token into every HTTP request."""

    def __init__(self, token_provider: OpenSkyTokenProvider) -> None:
        self._token_provider = token_provider

    def __call__(
        self,
        request: requests.PreparedRequest,
    ) -> requests.PreparedRequest:
        request.headers["Authorization"] = (
            f"Bearer {self._token_provider.get_token()}"
        )
        return request


class RefreshingJWTAuthentication(Authentication):
    """Trino authentication backed by an automatically refreshed JWT."""

    def __init__(self, token_provider: OpenSkyTokenProvider) -> None:
        self.token_provider = token_provider

    def set_http_session(
        self,
        http_session: requests.Session,
    ) -> requests.Session:
        http_session.auth = _RefreshingBearerAuth(self.token_provider)
        return http_session


def opensky_credentials_from_environment() -> tuple[str, str]:
    """Read the OpenSky website credentials required for Trino."""
    username = os.environ.get(OPENSKY_USERNAME_ENV, "").strip()
    password = os.environ.get(OPENSKY_PASSWORD_ENV, "")
    missing = [
        name
        for name, value in (
            (OPENSKY_USERNAME_ENV, username),
            (OPENSKY_PASSWORD_ENV, password),
        )
        if not value
    ]
    if missing:
        raise TrinoAuthError(
            "Automatic OpenSky Trino authentication requires "
            + " and ".join(missing)
            + " to be set"
        )
    return username.lower(), password


def rows_to_arrow(
    rows: list[tuple[Any, ...]],
    columns: list[str],
) -> pa.Table:
    """Convert one batch of Trino DB-API rows into an Arrow table."""
    if not rows:
        return pa.table({column: [] for column in columns})

    return pa.Table.from_arrays(
        [pa.array(values) for values in zip(*rows)],
        names=columns,
    )


def create_connection(user: str = "") -> Connection:
    """Create an automatically authenticated OpenSky Trino connection."""
    username, password = opensky_credentials_from_environment()
    requested_user = user.strip().lower()
    if requested_user and requested_user != username:
        raise TrinoAuthError(
            "The requested Trino user does not match OPENSKY_USERNAME"
        )

    authentication = RefreshingJWTAuthentication(
        OpenSkyTokenProvider(username, password)
    )
    return connect(
        host="trino.opensky-network.org",
        port=443,
        user=username,
        catalog="minio",
        schema="osky",
        http_scheme="https",
        auth=authentication,
        timezone="UTC",
        encoding=["json+zstd", "json"],
    )
