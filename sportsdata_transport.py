"""
sportsdata_transport.py
Centralized HTTP transport layer for SportsDataIO.

Supports:
    - live API
    - replay API
    - automatic key routing
    - future rate limiting hooks
"""

from __future__ import annotations

import os
import requests
from urllib.parse import urljoin

# -----------------------------
# Configuration
# -----------------------------

MODE = os.getenv("SPORTSDATA_MODE", "live").lower()

LIVE_BASE = "https://api.sportsdata.io"
REPLAY_BASE = "https://replay.sportsdata.io/api"

LIVE_KEY = os.getenv("SPORTSDATA_API_KEY", "")
REPLAY_KEY = os.getenv("SPORTSDATA_REPLAY_KEY", "")

TIMEOUT = float(os.getenv("SPORTSDATA_TIMEOUT", "25"))

# -----------------------------
# Helpers
# -----------------------------

def _base() -> str:
    return REPLAY_BASE if MODE == "replay" else LIVE_BASE


def _key() -> str:
    key = REPLAY_KEY if MODE == "replay" else LIVE_KEY
    if not key:
        raise RuntimeError("SportsData API key missing.")
    return key


def _headers() -> dict:
    return {
        "Ocp-Apim-Subscription-Key": _key(),
        "Accept": "application/json",
        "User-Agent": "covariant-engine/1.0"
    }


# -----------------------------
# Public API
# -----------------------------

def get(path: str, params: dict | None = None):
    """
    Example path:
        /v3/mlb/scores/json/GamesByDate/2023-AUG-15
    """

    if not path.startswith("/"):
        raise ValueError("Path must start with '/'")

    url = urljoin(_base(), path)

    r = requests.get(
        url,
        headers=_headers(),
        params=params,
        timeout=TIMEOUT
    )

    r.raise_for_status()
    return r.json()
