"""API-key generation and hashing.

A key looks like ``dfk_3f9a1c2b_<secret>``. Only its SHA-256 hash is stored, so a database
leak does not leak working keys. A fast hash is fine here (unlike passwords) because the
secret is 32 random bytes, so it cannot be brute-forced from the hash.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

KEY_PREFIX = "dfk"


@dataclass(frozen=True)
class NewApiKey:
    secret: str  # shown to the user exactly once
    prefix: str  # stored, shown in the UI to tell keys apart
    key_hash: str  # stored


def hash_api_key(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def generate_api_key() -> NewApiKey:
    public = secrets.token_hex(4)
    secret = f"{KEY_PREFIX}_{public}_{secrets.token_urlsafe(32)}"
    return NewApiKey(secret=secret, prefix=f"{KEY_PREFIX}_{public}", key_hash=hash_api_key(secret))
