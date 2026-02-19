import os
import requests
from datetime import datetime

REPLAY_KEY = os.getenv("SPORTSDATA_REPLAY_KEY")

META_URL = "https://replay.sportsdata.io/api/metadata"


def get_replay_time():
    """
    Returns replay-controlled datetime.
    """
    if not REPLAY_KEY:
        raise RuntimeError("SPORTSDATA_REPLAY_KEY not set")

    r = requests.get(
        META_URL,
        params={"key": REPLAY_KEY},
        timeout=20
    )
    r.raise_for_status()

    data = r.json()

    # Example field from replay metadata
    ts = data.get("CurrentDateTime")

    if not ts:
        raise RuntimeError("Replay metadata missing CurrentDateTime")

    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
