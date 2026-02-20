from __future__ import annotations

import json
import os
import time
import math
import random
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    Retry = None  # type: ignore


BASE_URL = "https://api.the-odds-api.com/v4/sports"


# =============================
# Errors
# =============================

class OddsClientError(RuntimeError):
    pass


class OddsAuthError(OddsClientError):
    pass


class OddsRateLimitError(OddsClientError):
    pass


class OddsHTTPError(OddsClientError):
    pass


class OddsParseError(OddsClientError):
    pass


# =============================
# Odds math helpers
# =============================

def implied_prob_from_decimal(decimal_odds: float) -> float:
    d = float(decimal_odds)
    if d <= 1.0:
        raise ValueError("Decimal odds must be > 1.0")
    return 1.0 / d


def decimal_from_american(american: float) -> float:
    a = float(american)
    if a == 0:
        raise ValueError("American odds cannot be 0")
    if a > 0:
        return 1.0 + (a / 100.0)
    return 1.0 + (100.0 / abs(a))


def american_from_decimal(decimal_odds: float) -> int:
    d = float(decimal_odds)
    if d <= 1.0:
        raise ValueError("Decimal odds must be > 1.0")
    if d >= 2.0:
        return int(round((d - 1.0) * 100))
    return int(round(-100.0 / (d - 1.0)))


def normalize_name(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def remove_overround(probs: Mapping[str, float]) -> Dict[str, float]:
    """
    Normalize implied probabilities so they sum to 1.0 (simple vig removal).
    """
    total = sum(float(p) for p in probs.values())
    if total <= 0:
        return {k: 0.0 for k in probs}
    return {k: float(p) / total for k, p in probs.items()}


def expected_value_win(prob: float, decimal_odds: float) -> float:
    """
    EV per $1 stake for a simple win bet:
      EV = p*(odds-1) - (1-p)*1
    """
    p = clamp(float(prob), 0.0, 1.0)
    d = float(decimal_odds)
    return p * (d - 1.0) - (1.0 - p)


# =============================
# Optional cache + SWR
# =============================

@dataclass
class CacheConfig:
    enabled: bool = True
    ttl_seconds: int = 30
    dir_path: str = ".cache/odds_api"
    stale_while_revalidate_seconds: int = 0
    compress: bool = False  # keep False in Termux for speed

    def path_for(self, key: str) -> Path:
        p = Path(self.dir_path)
        p.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return p / f"{h}.json"

    def read(self, key: str) -> Optional[Tuple[float, Any]]:
        if not self.enabled:
            return None
        p = self.path_for(key)
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            ts = float(raw.get("_cached_at", 0))
            return ts, raw.get("data")
        except Exception:
            return None

    def write(self, key: str, data: Any) -> None:
        if not self.enabled:
            return
        p = self.path_for(key)
        try:
            payload = {"_cached_at": time.time(), "data": data}
            p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


# =============================
# Data model (lightweight)
# =============================

@dataclass(frozen=True)
class Outcome:
    name: str
    price: Optional[float] = None
    point: Optional[float] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class Market:
    key: str
    outcomes: Tuple[Outcome, ...]


@dataclass(frozen=True)
class Bookmaker:
    key: str
    title: str
    last_update: Optional[str]
    markets: Tuple[Market, ...]


@dataclass(frozen=True)
class Event:
    id: str
    commence_time: Optional[str]
    home_team: Optional[str]
    away_team: Optional[str]
    bookmakers: Tuple[Bookmaker, ...]


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def parse_events(raw: Any) -> List[Event]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise OddsParseError(f"Expected list from API, got {type(raw)}")
    out: List[Event] = []
    for g in raw:
        if not isinstance(g, dict):
            continue
        books: List[Bookmaker] = []
        for b in (g.get("bookmakers") or []):
            if not isinstance(b, dict):
                continue
            markets: List[Market] = []
            for m in (b.get("markets") or []):
                if not isinstance(m, dict):
                    continue
                outcomes: List[Outcome] = []
                for o in (m.get("outcomes") or []):
                    if not isinstance(o, dict):
                        continue
                    outcomes.append(
                        Outcome(
                            name=str(o.get("name") or "").strip(),
                            price=_as_float(o.get("price")),
                            point=_as_float(o.get("point")),
                            description=(str(o.get("description")).strip() if o.get("description") else None),
                        )
                    )
                mk = str(m.get("key") or "").strip()
                if mk and outcomes:
                    markets.append(Market(key=mk, outcomes=tuple(outcomes)))
            books.append(
                Bookmaker(
                    key=str(b.get("key") or "").strip(),
                    title=str(b.get("title") or "").strip(),
                    last_update=(str(b.get("last_update")).strip() if b.get("last_update") else None),
                    markets=tuple(markets),
                )
            )
        out.append(
            Event(
                id=str(g.get("id") or "").strip(),
                commence_time=(str(g.get("commence_time")).strip() if g.get("commence_time") else None),
                home_team=(str(g.get("home_team")).strip() if g.get("home_team") else None),
                away_team=(str(g.get("away_team")).strip() if g.get("away_team") else None),
                bookmakers=tuple(books),
            )
        )
    return out


# =============================
# Local pacing (soft rate limit)
# =============================

@dataclass
class RateConfig:
    enabled: bool = True
    min_interval_seconds: float = 0.20  # don't make calls faster than 5/sec locally


# =============================
# Client
# =============================

@dataclass
class OddsClient:
    api_key: Optional[str] = None
    base_url: str = BASE_URL
    timeout_seconds: int = 25

    # retry + backoff
    max_attempts: int = 6
    backoff_base: float = 0.6
    backoff_max: float = 8.0
    jitter: float = 0.25

    # urllib3 retry adapter (optional)
    urllib3_retries: int = 2

    cache: CacheConfig = field(default_factory=CacheConfig)
    rate: RateConfig = field(default_factory=RateConfig)
    user_agent: str = "covariant/odds-client (requests)"
    debug: bool = False

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
        if not self.api_key:
            raise OddsAuthError("ODDS_API_KEY (or THE_ODDS_API_KEY) not set")

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

        if Retry is not None and self.urllib3_retries > 0:
            retry = Retry(
                total=self.urllib3_retries,
                connect=self.urllib3_retries,
                read=self.urllib3_retries,
                status=self.urllib3_retries,
                backoff_factor=0.2,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset(["GET"]),
                raise_on_status=False,
                respect_retry_after_header=True,
            )
            adapter = HTTPAdapter(max_retries=retry)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

        self.last_headers: Dict[str, str] = {}
        self._last_call_ts: float = 0.0

    # ---- internal ----

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[odds_client] {msg}")

    def _pace(self) -> None:
        if not self.rate.enabled:
            return
        now = time.time()
        dt = now - self._last_call_ts
        if dt < self.rate.min_interval_seconds:
            time.sleep(self.rate.min_interval_seconds - dt)
        self._last_call_ts = time.time()

    def _cache_key(self, path: str, params: Mapping[str, Any]) -> str:
        items = sorted((str(k), str(v)) for k, v in params.items())
        blob = json.dumps(items, separators=(",", ":"), ensure_ascii=True)
        return f"{path}|{blob}"

    def _headers_snapshot(self, r: requests.Response) -> Dict[str, str]:
        return {k.lower(): v for k, v in (r.headers or {}).items()}

    def _extract_retry_after(self) -> Optional[float]:
        ra = self.last_headers.get("retry-after")
        if not ra:
            return None
        try:
            return float(ra)
        except Exception:
            return None

    def _request_json(self, path: str, params: Mapping[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        full_params = dict(params)
        full_params["apiKey"] = self.api_key

        ck = self._cache_key(path, full_params)
        cached = self.cache.read(ck)
        if cached is not None:
            ts, data = cached
            age = time.time() - ts
            if age <= self.cache.ttl_seconds:
                self._log(f"cache hit fresh age={age:.1f}s path={path}")
                return data
            if self.cache.stale_while_revalidate_seconds and age <= (self.cache.ttl_seconds + self.cache.stale_while_revalidate_seconds):
                # stale but acceptable: return stale immediately; caller can choose to refresh elsewhere
                self._log(f"cache hit stale age={age:.1f}s path={path}")
                return data

        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            self._pace()

            try:
                r = self.session.get(url, params=full_params, timeout=self.timeout_seconds)
                self.last_headers = self._headers_snapshot(r)

                if r.status_code in (401, 403):
                    raise OddsAuthError(f"Odds API auth failed ({r.status_code}). Check ODDS_API_KEY.")

                if r.status_code == 429:
                    ra = self._extract_retry_after()
                    remaining = self.last_headers.get("x-requests-remaining")
                    used = self.last_headers.get("x-requests-used")
                    msg = f"Odds API 429 rate limit. retry-after={ra} remaining={remaining} used={used}"
                    # If last attempt, raise
                    if attempt >= self.max_attempts:
                        raise OddsRateLimitError(msg)
                    # Sleep and retry with backoff
                    sleep_s = ra if ra is not None else self._compute_backoff(attempt)
                    self._log(f"{msg} -> sleeping {sleep_s:.2f}s (attempt {attempt}/{self.max_attempts})")
                    time.sleep(sleep_s)
                    continue

                if not r.ok:
                    snippet = (r.text or "")[:600]
                    if attempt >= self.max_attempts:
                        raise OddsHTTPError(f"Odds API HTTP {r.status_code}: {snippet}")
                    sleep_s = self._compute_backoff(attempt)
                    self._log(f"HTTP {r.status_code} -> sleeping {sleep_s:.2f}s then retry (attempt {attempt}/{self.max_attempts})")
                    time.sleep(sleep_s)
                    continue

                data = r.json()
                self.cache.write(ck, data)
                return data

            except (OddsAuthError,):  # do not retry auth
                raise
            except Exception as e:
                last_exc = e
                if attempt >= self.max_attempts:
                    raise OddsHTTPError(f"Request failed after {self.max_attempts} attempts: {e}") from e
                sleep_s = self._compute_backoff(attempt)
                self._log(f"network/parse error: {e} -> sleeping {sleep_s:.2f}s retry")
                time.sleep(sleep_s)

        # should not reach
        raise OddsHTTPError(f"Request failed: {last_exc}")

    def _compute_backoff(self, attempt: int) -> float:
        # exponential backoff with jitter
        base = self.backoff_base * (2 ** (attempt - 1))
        base = min(base, self.backoff_max)
        j = random.uniform(-self.jitter, self.jitter) * base
        return max(0.0, base + j)

    # ---- public ----

    def usage(self) -> Dict[str, Optional[str]]:
        return {
            "requests_remaining": self.last_headers.get("x-requests-remaining"),
            "requests_used": self.last_headers.get("x-requests-used"),
            "retry_after": self.last_headers.get("retry-after"),
        }

    def list_sports(self, all_sports: bool = False) -> Any:
        return self._request_json("/", params={"all": "true" if all_sports else "false"})

    def get_odds(
        self,
        sport_key: str,
        markets: Sequence[str],
        regions: str = "us",
        odds_format: str = "decimal",
        date_format: str = "iso",
        bookmakers: Optional[Sequence[str]] = None,
    ) -> Any:
        params: Dict[str, Any] = {
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        return self._request_json(f"/{sport_key}/odds", params=params)

    def get_event_odds(
        self,
        sport_key: str,
        event_id: str,
        markets: Sequence[str],
        regions: str = "us",
        odds_format: str = "decimal",
        date_format: str = "iso",
        bookmakers: Optional[Sequence[str]] = None,
    ) -> Any:
        params: Dict[str, Any] = {
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        return self._request_json(f"/{sport_key}/events/{event_id}/odds", params=params)

    # --- typed wrappers ---

    def get_odds_events(
        self,
        sport_key: str,
        markets: Sequence[str],
        regions: str = "us",
        odds_format: str = "decimal",
        date_format: str = "iso",
        bookmakers: Optional[Sequence[str]] = None,
    ) -> List[Event]:
        return parse_events(self.get_odds(sport_key, markets, regions, odds_format, date_format, bookmakers))

    def get_event_odds_events(
        self,
        sport_key: str,
        event_id: str,
        markets: Sequence[str],
        regions: str = "us",
        odds_format: str = "decimal",
        date_format: str = "iso",
        bookmakers: Optional[Sequence[str]] = None,
    ) -> List[Event]:
        raw = self.get_event_odds(sport_key, event_id, markets, regions, odds_format, date_format, bookmakers)
        # event endpoint returns a single event object or list depending on API tier; normalize to list
        if isinstance(raw, dict):
            raw = [raw]
        return parse_events(raw)

    # --- NBA convenience ---

    def get_nba_featured_markets_decimal(
        self,
        markets: Sequence[str] = ("h2h", "spreads", "totals"),
        regions: str = "us",
        bookmakers: Optional[Sequence[str]] = None,
    ) -> Any:
        return self.get_odds("basketball_nba", markets, regions, "decimal", "iso", bookmakers)

    def get_nba_h2h_decimal(self, regions: str = "us", bookmakers: Optional[Sequence[str]] = None) -> Any:
        return self.get_nba_featured_markets_decimal(("h2h",), regions, bookmakers)

    def get_nba_event_odds_decimal(
        self,
        event_id: str,
        markets: Sequence[str],
        regions: str = "us",
        bookmakers: Optional[Sequence[str]] = None,
    ) -> Any:
        return self.get_event_odds("basketball_nba", event_id, markets, regions, "decimal", "iso", bookmakers)


# =============================
# Extraction utilities (typed + advanced)
# =============================

def iter_markets(book: Bookmaker, market_key: str) -> Iterator[Market]:
    for m in book.markets:
        if m.key == market_key:
            yield m


def select_bookmaker(
    event: Union[Mapping[str, Any], Event],
    prefer_keys: Sequence[str] = (),
    prefer_titles: Sequence[str] = (),
) -> Optional[Union[Mapping[str, Any], Bookmaker]]:
    """
    Works on raw dict events or typed Event.
    """
    keys_norm = [normalize_name(k) for k in prefer_keys]
    titles_norm = [normalize_name(t) for t in prefer_titles]

    if isinstance(event, Event):
        books = list(event.bookmakers)
        if not books:
            return None
        for b in books:
            k = normalize_name(b.key)
            t = normalize_name(b.title)
            if k and k in keys_norm:
                return b
            if t and t in titles_norm:
                return b
        return books[0]

    # dict fallback
    books = [b for b in (event.get("bookmakers") or []) if isinstance(b, dict)]
    if not books:
        return None
    for b in books:
        k = normalize_name(str(b.get("key", "")))
        t = normalize_name(str(b.get("title", "")))
        if k and k in keys_norm:
            return b
        if t and t in titles_norm:
            return b
    return books[0]


def extract_h2h_prices(
    data: Sequence[Mapping[str, Any]],
    prefer_bookmakers: Sequence[str] = ("draftkings", "fanduel", "betmgm", "caesars", "pointsbetus"),
    best_across_books: bool = False,
) -> Dict[str, float]:
    """
    Raw-dict extractor: team -> best price (decimal)
    """
    prices: Dict[str, float] = {}

    for game in (data or []):
        if not isinstance(game, dict):
            continue

        if best_across_books:
            best: Dict[str, float] = {}
            for b in (game.get("bookmakers") or []):
                if not isinstance(b, dict):
                    continue
                for m in (b.get("markets") or []):
                    if not isinstance(m, dict) or m.get("key") != "h2h":
                        continue
                    for o in (m.get("outcomes") or []):
                        if not isinstance(o, dict):
                            continue
                        name = o.get("name")
                        price = o.get("price")
                        if not name or price is None:
                            continue
                        try:
                            p = float(price)
                        except Exception:
                            continue
                        key = str(name).strip()
                        if key not in best or p > best[key]:
                            best[key] = p
            prices.update(best)
        else:
            b = select_bookmaker(game, prefer_keys=prefer_bookmakers)
            if not isinstance(b, dict):
                continue
            for m in (b.get("markets") or []):
                if not isinstance(m, dict) or m.get("key") != "h2h":
                    continue
                for o in (m.get("outcomes") or []):
                    if not isinstance(o, dict):
                        continue
                    name = o.get("name")
                    price = o.get("price")
                    if not name or price is None:
                        continue
                    try:
                        prices[str(name).strip()] = float(price)
                    except Exception:
                        pass

    return prices


def best_h2h_per_event(
    events: Sequence[Event],
    prefer_bookmakers: Sequence[str] = ("draftkings", "fanduel", "betmgm", "caesars"),
) -> List[Dict[str, Any]]:
    """
    Typed extractor: per-event best line (by preferred book, fallback first)
    Returns rows with implied probs and vig-removed probs.
    """
    rows: List[Dict[str, Any]] = []

    for e in events:
        b = select_bookmaker(e, prefer_keys=prefer_bookmakers)
        if not isinstance(b, Bookmaker):
            continue

        market = next(iter_markets(b, "h2h"), None)
        if not market:
            continue

        prices: Dict[str, float] = {}
        for o in market.outcomes:
            if o.name and o.price:
                prices[o.name] = float(o.price)

        if len(prices) < 2:
            continue

        implied = {k: implied_prob_from_decimal(v) for k, v in prices.items()}
        fair = remove_overround(implied)

        rows.append({
            "event_id": e.id,
            "commence_time": e.commence_time,
            "home_team": e.home_team,
            "away_team": e.away_team,
            "bookmaker": b.key or b.title,
            "prices_decimal": prices,
            "implied_probs": implied,
            "fair_probs": fair,
            "overround": sum(implied.values()),
        })

    return rows


def consensus_h2h(
    events: Sequence[Event],
    market_key: str = "h2h",
) -> List[Dict[str, Any]]:
    """
    Consensus line across all books (median implied prob per team).
    Returns fair probs (vig removed) for each event.
    """
    out: List[Dict[str, Any]] = []

    for e in events:
        # team -> list of implied probs across books
        buckets: Dict[str, List[float]] = {}

        for b in e.bookmakers:
            mk = next(iter_markets(b, market_key), None)
            if not mk:
                continue
            for o in mk.outcomes:
                if not o.name or not o.price:
                    continue
                try:
                    ip = implied_prob_from_decimal(float(o.price))
                except Exception:
                    continue
                buckets.setdefault(o.name, []).append(ip)

        if len(buckets) < 2:
            continue

        # median implied per team
        med: Dict[str, float] = {}
        for team, vals in buckets.items():
            vals2 = sorted(vals)
            n = len(vals2)
            if n % 2 == 1:
                med[team] = vals2[n // 2]
            else:
                med[team] = 0.5 * (vals2[n // 2 - 1] + vals2[n // 2])

        fair = remove_overround(med)

        out.append({
            "event_id": e.id,
            "home_team": e.home_team,
            "away_team": e.away_team,
            "median_implied": med,
            "fair_probs": fair,
            "overround_median": sum(med.values()),
            "books_count": len(e.bookmakers),
        })

    return out


# =============================
# Backwards-compatible drop-in functions
# =============================

_default_client: Optional[OddsClient] = None


def _client() -> OddsClient:
    global _default_client
    if _default_client is None:
        _default_client = OddsClient()
    return _default_client


def get_nba_featured_markets_decimal(markets=("h2h", "spreads", "totals"), regions="us"):
    return _client().get_nba_featured_markets_decimal(markets=markets, regions=regions)


def get_nba_h2h_decimal(regions="us"):
    return _client().get_nba_h2h_decimal(regions=regions)


def get_nba_event_odds_decimal(event_id, markets, regions="us"):
    return _client().get_nba_event_odds_decimal(event_id=event_id, markets=markets, regions=regions)


def extract_h2h_prices_first_bookmaker(data):
    # historical behavior: first bookmaker only (raw dicts)
    return extract_h2h_prices(data, prefer_bookmakers=(), best_across_books=False)
