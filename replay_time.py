"""
Replay Time Controller
----------------------
Makes Covariant operate on replay timeline instead of real UTC.
"""

from datetime import datetime, timezone
from sportsdata_client import get


def get_replay_metadata():
    """
    Returns replay metadata including simulated current time.
    """
    return get("/api/metadata")


def get_replay_time() -> datetime:
    """
    Returns replay 'now' as timezone-aware datetime.
    """
    meta = get_replay_metadata()

    # SportsData returns ISO timestamp
    ts = (
        meta.get("CurrentDateTime")
        or meta.get("currentDateTime")
        or meta.get("DateTime")
    )

    if not ts:
        raise RuntimeError("Replay metadata missing CurrentDateTime")

    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc)


def replay_date_string():
    """
    Returns YYYY-MMM-DD format required by MLB endpoints.
    Example: 2023-AUG-15
    """
    dt = get_replay_time()
    return dt.strftime("%Y-%b-%d").upper()
