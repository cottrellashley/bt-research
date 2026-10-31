"""Snowflake connection for ``SQLLoader`` (``pip install snowflake-connector-python``).

Credentials never live in code or manifests: the connection reads ``SNOWFLAKE_ACCOUNT``,
``SNOWFLAKE_USER``, ``SNOWFLAKE_ROLE``, ``SNOWFLAKE_WAREHOUSE``, ``SNOWFLAKE_DATABASE``,
``SNOWFLAKE_SCHEMA`` and either ``SNOWFLAKE_PRIVATE_KEY_PATH`` (key-pair, for service users) or
``SNOWFLAKE_AUTHENTICATOR`` (``externalbrowser`` for SSO; the default). Read-only access is enough:
loaders only run SELECTs.
"""

from __future__ import annotations

import os

from nautilus_fi.data.loaders.base import SQLLoader


def snowflake_connect(**overrides):
    try:
        import snowflake.connector as sf
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("Snowflake loader needs snowflake-connector-python (pip install 'nautilus-fi[snowflake]')") from e
    kw = {"account": os.environ.get("SNOWFLAKE_ACCOUNT"), "user": os.environ.get("SNOWFLAKE_USER"), "role": os.environ.get("SNOWFLAKE_ROLE"), "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"), "database": os.environ.get("SNOWFLAKE_DATABASE"), "schema": os.environ.get("SNOWFLAKE_SCHEMA")}
    key_path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH")
    if key_path:
        from cryptography.hazmat.primitives import serialization

        pwd = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
        with open(key_path, "rb") as f:
            key = serialization.load_pem_private_key(f.read(), password=pwd.encode() if pwd else None)
        kw["private_key"] = key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    else:
        kw["authenticator"] = os.environ.get("SNOWFLAKE_AUTHENTICATOR", "externalbrowser")
    kw.update(overrides)
    missing = [k for k in ("account", "user") if not kw.get(k)]
    if missing:
        raise ValueError(f"Snowflake connection needs {missing} (environment variables SNOWFLAKE_ACCOUNT / SNOWFLAKE_USER)")
    return sf.connect(**{k: v for k, v in kw.items() if v is not None})


def snowflake_loader(chunk_rows: int = 250_000, **overrides) -> SQLLoader:
    return SQLLoader(lambda: snowflake_connect(**overrides), paramstyle="pyformat", chunk_rows=chunk_rows)


__all__ = ["snowflake_connect", "snowflake_loader"]
