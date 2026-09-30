"""Shared OpenSky Trino connection and Arrow helpers."""

import json
import os
from pathlib import Path
import sys
from typing import Any

import pyarrow as pa
from trino.auth import OAuth2Authentication
from trino.dbapi import Connection, connect
from trino.exceptions import TrinoAuthError


BATCH_SIZE = 100_000
DEFAULT_TRINO_USER = os.environ.get("OPENSKY_TRINO_USER", "goodejonah")
TOKEN_CACHE_PATH_ENV = "OPENSKY_TRINO_TOKEN_CACHE"
TOKEN_CACHE_SERVICE = "planequery-opensky-trino"


def default_token_cache_path() -> Path:
    """Return the fallback OAuth token path for the current platform."""
    configured = os.environ.get(TOKEN_CACHE_PATH_ENV)
    if configured:
        return Path(configured).expanduser()
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "PlaneQuery"
            / "opensky-trino-oauth.json"
        )
    state_root = Path(
        os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
    )
    return state_root / "planequery" / "opensky-trino-oauth.json"


def _load_keyring() -> Any | None:
    try:
        import keyring
    except ImportError:
        return None
    return keyring


class PersistentOAuthTokenCache:
    """Persist OAuth tokens in the OS keyring or a mode-0600 JSON file."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        keyring_module: Any | None = None,
        load_system_keyring: bool = True,
    ) -> None:
        self.path = path or default_token_cache_path()
        self._keyring = (
            _load_keyring() if load_system_keyring else keyring_module
        )
        if keyring_module is not None:
            self._keyring = keyring_module
        self._reported_file_fallback = False

    @staticmethod
    def _account(key: str | None) -> str:
        return key or "default"

    def _get_from_keyring(self, key: str | None) -> str | None:
        if self._keyring is None:
            return None
        try:
            return self._keyring.get_password(
                TOKEN_CACHE_SERVICE,
                self._account(key),
            )
        except Exception:
            self._keyring = None
            return None

    def _store_in_keyring(self, key: str | None, token: str) -> bool:
        if self._keyring is None:
            return False
        try:
            self._keyring.set_password(
                TOKEN_CACHE_SERVICE,
                self._account(key),
                token,
            )
        except Exception:
            self._keyring = None
            return False
        return True

    def _read_file(self) -> dict[str, str]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): value
            for key, value in payload.items()
            if isinstance(value, str)
        }

    def _store_in_file(self, key: str | None, token: str) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tokens = self._read_file()
        tokens[self._account(key)] = token
        temporary_path = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.tmp"
        )
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(tokens, file)
                file.write("\n")
            os.replace(temporary_path, self.path)
            self.path.chmod(0o600)
        finally:
            temporary_path.unlink(missing_ok=True)
        if not self._reported_file_fallback:
            print(
                "System keyring unavailable; cached the OpenSky OAuth token "
                f"in {self.path} with mode 0600.",
                file=sys.stderr,
            )
            self._reported_file_fallback = True

    def get_token_from_cache(self, key: str | None) -> str | None:
        token = self._get_from_keyring(key)
        if token is not None:
            return token
        return self._read_file().get(self._account(key))

    def store_token_to_cache(self, key: str | None, token: str) -> None:
        if not self._store_in_keyring(key, token):
            self._store_in_file(key, token)


def wait_for_external_authentication(url: str) -> None:
    """Pause Trino's OAuth polling until browser authentication is complete.

    The Trino Python client otherwise starts polling immediately and gives up
    after five attempts, which is usually too short for a copied browser URL on
    a headless or remote machine.
    """
    print(
        "Open this URL in a browser and sign in with the same OpenSky account:\n"
        f"{url}",
        flush=True,
    )
    try:
        input("After authentication succeeds in the browser, press Enter here: ")
    except EOFError as error:
        raise TrinoAuthError(
            "OpenSky external authentication requires an interactive terminal. "
            "Run the downloader from a terminal, open the printed URL, then "
            "press Enter after the browser login succeeds."
        ) from error


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


def create_connection(user: str) -> Connection:
    """Create an external OAuth-authenticated OpenSky Trino connection."""
    authentication = OAuth2Authentication(
        redirect_auth_url_handler=wait_for_external_authentication,
    )
    # trino 0.340 has no public token-cache injection point. It is pinned in
    # pyproject.toml, so replace its internal in-memory/keyring-only cache with
    # the cross-platform cache above.
    authentication._bearer._token_cache = PersistentOAuthTokenCache()
    return connect(
        host="trino.opensky-network.org",
        port=443,
        user=user.lower(),
        catalog="minio",
        schema="osky",
        http_scheme="https",
        auth=authentication,
        timezone="UTC",
        encoding=["json+zstd", "json"],
    )
