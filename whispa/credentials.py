"""The optional AI server API key, kept in the system's credential store (Windows Credential
Manager, macOS Keychain, the Linux Secret Service) instead of in config.json.

If no credential store is available, the key stays in config.json (which only you can read)
and these return False / "".
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

SERVICE = "Murmr"
ACCOUNT = "llm_api_key"
_stored: str | None = None  # what we last wrote, so saving settings doesn't rewrite it


def store_api_key(key: str) -> bool:
    global _stored
    if key == _stored:
        return True
    try:
        import keyring

        keyring.set_password(SERVICE, ACCOUNT, key)
    except Exception as exc:  # noqa: BLE001 - no backend, locked keychain...
        log.warning("Couldn't keep the API key in the system credential store: %s", exc)
        return False
    _stored = key
    return True


def load_api_key() -> str:
    global _stored
    try:
        import keyring

        key = keyring.get_password(SERVICE, ACCOUNT) or ""
    except Exception as exc:  # noqa: BLE001
        log.warning("Couldn't read the API key from the system credential store: %s", exc)
        return ""
    _stored = key or None
    return key


def holds_key() -> bool:
    """A key was read from (or written to) the store in this run. False after a failed read,
    so a keychain that's briefly unavailable never gets the key deleted."""
    return bool(_stored)


def delete_api_key() -> None:
    global _stored
    try:
        import keyring

        keyring.delete_password(SERVICE, ACCOUNT)
    except Exception:  # noqa: BLE001 - nothing stored
        pass
    _stored = None
