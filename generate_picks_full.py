
from __future__ import annotations
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent / ".env")
from sportsdata_client import get

from env_loader import load_env
load_env()



# --- SportsDataIO client (live/replay) ---
import json as sdio_json
#!/usr/bin/env python3

from replay_clock import get_replay_time
import os
from dotenv import load_dotenv
load_dotenv()
import argparse
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import platform
import random
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import generate_picks as gp  # must provide fetch_odds()


RUNS_DIR = Path("runs")


# -----------------------------
# Utilities
# -----------------------------
def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _utc_iso() -> str:
    # timezone-aware UTC timestamp; avoids deprecated utcnow()
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _coerce_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _coerce_american(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        # accepts "781", 781, 781.0, "+120", "-105"
        s = str(x).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _amer_to_dec(a: int) -> float:
    a = int(a)
    if a == 0:
        return 0.0
    return 1.0 + (a / 100.0) if a > 0 else 1.0 + (100.0 / abs(a))


def _dec_to_imp(dec: float) -> float:
    dec = float(dec)
    if dec <= 1.0:
        return 0.0
    return 1.0 / dec


def _clamp01(p: float) -> float:
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _trimmed(xs: List[float], trim: float) -> List[float]:
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x)) and not math.isinf(float(x))]
    xs.sort()
    if not xs:
        return []
    if trim <= 0:
        return xs
    k = int(len(xs) * trim)
    if len(xs) - 2 * k <= 0:
        return xs
    return xs[k: len(xs) - k]


def _trimmed_mean(xs: List[float], trim: float) -> float:
    ys = _trimmed(xs, trim)
    if not ys:
        return float("nan")
    return sum(ys) / len(ys)


def _winsorized_mean(xs: List[float], trim: float) -> float:
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x)) and not math.isinf(float(x))]
    xs.sort()
    if not xs:
        return float("nan")
    if trim <= 0:
        return sum(xs) / len(xs)
    k = int(len(xs) * trim)
    if k <= 0:
        return sum(xs) / len(xs)
    lo = xs[k]
    hi = xs[-k - 1]
    ys = [ _clip(x, lo, hi) for x in xs ]
    return sum(ys) / len(ys)


def _weighted_median(vals: List[float], wts: List[float]) -> float:
    pairs = [(float(v), float(w)) for v, w in zip(vals, wts) if v is not None and w is not None and w > 0]
    if not pairs:
        return float("nan")
    pairs.sort(key=lambda t: t[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return float("nan")
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= 0.5 * total:
            return float(v)
    return float(pairs[-1][0])


def _normalize_name(s: str) -> str:
    # Aggressive normalization to match outcomes across books
    s = (s or "").strip().lower()
    s = " ".join(s.split())

    # unify unicode apostrophes
    s = s.replace("’", "'")

    # remove punctuation-ish stuff
    s = re.sub(r"[.,:;()\[\]{}]", "", s)

    # collapse multiple spaces again
    s = " ".join(s.split())

    return s


def _normalize_book(s: str) -> str:
    s = (s or "").strip().lower()
    s = " ".join(s.split())
    # common aliases
    alias = {
        "william hill us": "williamhill",
        "william hill": "williamhill",
        "dk": "draftkings",
        "draft kings": "draftkings",
        "fd": "fanduel",
        "fan duel": "fanduel",
        "bet mgm": "betmgm",
        "mgm": "betmgm",
        "caesars sportsbook": "caesars",
        "pointsbetus": "pointsbet",
    }
    return alias.get(s, s)


def _normalize_market_type(s: str) -> str:
    s = (s or "").strip().lower()
    s = " ".join(s.split())
    # unify common market keys
    alias = {
        "h2h": "moneyline",
        "ml": "moneyline",
        "moneyline": "moneyline",
        "spreads": "spread",
        "spread": "spread",
        "totals": "total",
        "total": "total",
        "over_under": "total",
        "over/under": "total",
        "ou": "total",
        "player_points": "player_points",
        "player_rebounds": "player_rebounds",
        "player_assists": "player_assists",
    }
    return alias.get(s, s)


def _format_line(line: Optional[float]) -> str:
    if line is None:
        return ""
    # keep stable representation
    # avoid "-0.0"
    if abs(line) < 1e-12:
        line = 0.0
    # 0.5 -> "0.5", 221.5 -> "221.5"
    if float(line).is_integer():
        return str(int(line))
    return f"{line:.3f}".rstrip("0").rstrip(".")


def _pick_outcome_name(p: dict) -> str:
    return str(p.get("selection") or p.get("name") or p.get("team") or p.get("outcome") or "")


def _ensure_model_prob(p: dict) -> Optional[float]:
    v = p.get("model_prob")
    if isinstance(v, (int, float)) and 0.0 < float(v) < 1.0:
        return float(v)
    for k in ("p_model", "p", "win_prob", "prob", "model_p"):
        v = p.get(k)
        if isinstance(v, (int, float)) and 0.0 < float(v) < 1.0:
            return float(v)
    return None


def _pick_line(p: dict) -> Optional[float]:
    # common keys
    for k in ("line", "point", "handicap", "total", "strike", "threshold"):
        v = _safe_float(p.get(k))
        if v is not None:
            return v
    return None


def _guess_over_under_name(raw: str) -> str:
    s = _normalize_name(raw)
    if s in ("over", "o"):
        return "over"
    if s in ("under", "u"):
        return "under"
    return raw


# -----------------------------
# Market tape representation
# -----------------------------
@dataclass(frozen=True)
class TapeRow:
    event_id: str
    market_type: str
    line: Optional[float]          # spread/total points etc, None for ML
    outcome_name: str              # raw-ish outcome
    outcome_key: str               # normalized outcome name
    outcome_id: str                # normalized outcome + line tag
    book: str
    decimal: float

    @property
    def implied(self) -> float:
        return _dec_to_imp(self.decimal)


def _infer_event_id(ev: dict) -> str:
    return str(ev.get("id") or ev.get("event_id") or ev.get("eventId") or ev.get("key") or ev.get("event_key") or "")


def _infer_book(bm: dict) -> str:
    return str(bm.get("title") or bm.get("key") or bm.get("name") or bm.get("book") or "unknown")


def _infer_market_type(m: dict, fallback: str = "unknown") -> str:
    return str(m.get("key") or m.get("market") or m.get("type") or m.get("name") or fallback)


def _infer_outcome_name(o: dict) -> str:
    return str(o.get("name") or o.get("outcome") or o.get("team") or o.get("label") or "")


def _infer_outcome_decimal(o: dict) -> Optional[float]:
    # direct decimal
    dec = _safe_float(o.get("decimal"))
    if dec and dec > 1.0:
        return dec

    # sometimes "price" is decimal
    dec2 = _safe_float(o.get("price"))
    if dec2 and dec2 > 1.0:
        return dec2

    # sometimes "odds"/"american"/"price" is american
    amer = _coerce_american(o.get("american") or o.get("odds") or o.get("price"))
    if amer is not None:
        dec3 = _amer_to_dec(amer)
        if dec3 > 1.0:
            return dec3

    return None


def _infer_outcome_line(o: dict) -> Optional[float]:
    # TheOddsAPI typically uses "point" for spreads/totals outcomes.
    # Many adapters use: point, handicap, total, line
    for k in ("point", "handicap", "total", "line"):
        v = _safe_float(o.get(k))
        if v is not None:
            return v
    return None


def _make_outcome_id(outcome_key: str, line: Optional[float]) -> str:
    # Line is part of identity for totals/spreads.
    # For ML or prop without line, line is empty.
    tag = _format_line(line)
    return f"{outcome_key}::{tag}"


def _extract_market_rows(odds_blob: Any) -> List[TapeRow]:
    """
    Normalize gp.fetch_odds() output into a tape of rows:
      (event_id, market_type, line, outcome_name, book, decimal)

    Supports:
      - TheOddsAPI-ish: events -> bookmakers -> markets -> outcomes
      - provider adapters that already produce row dicts
      - nested dict market maps
    """
    if not odds_blob:
        return []

    data = odds_blob
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if isinstance(data, dict) and "events" in data:
        data = data["events"]

    rows: List[TapeRow] = []

    # Already row-like dicts?
    if isinstance(data, list) and data and isinstance(data[0], dict) and ("event_id" in data[0] and "market_type" in data[0]):
        for r in data:
            try:
                event_id = str(r.get("event_id", ""))
                market_type = _normalize_market_type(str(r.get("market_type", "")))
                outcome_name = str(r.get("outcome_name") or r.get("name") or "")
                book = _normalize_book(str(r.get("book", "unknown")))
                line = _safe_float(r.get("line") or r.get("point") or r.get("handicap") or r.get("total"))

                dec = _safe_float(r.get("decimal"))
                if not dec:
                    amer = _coerce_american(r.get("american") or r.get("odds") or r.get("price"))
                    if amer is not None:
                        dec = _amer_to_dec(amer)

                if event_id and market_type and outcome_name and dec and dec > 1.0:
                    ok = _normalize_name(_guess_over_under_name(outcome_name))
                    oid = _make_outcome_id(ok, line)
                    rows.append(TapeRow(
                        event_id=event_id,
                        market_type=market_type,
                        line=line,
                        outcome_name=outcome_name,
                        outcome_key=ok,
                        outcome_id=oid,
                        book=book,
                        decimal=float(dec),
                    ))
            except Exception:
                continue
        return rows

    if not isinstance(data, list):
        return rows

    for ev in data:
        if not isinstance(ev, dict):
            continue
        event_id = _infer_event_id(ev)
        if not event_id:
            continue

        bookmakers = ev.get("bookmakers") or ev.get("books") or ev.get("sites") or []
        if isinstance(bookmakers, dict):
            bookmakers = [bookmakers]
        if not isinstance(bookmakers, list):
            continue

        for bm in bookmakers:
            if not isinstance(bm, dict):
                continue
            book = _normalize_book(_infer_book(bm))

            markets = bm.get("markets") or bm.get("odds") or bm.get("lines") or []
            # dict-of-markets => convert to list
            if isinstance(markets, dict):
                tmp = []
                for mk, mv in markets.items():
                    if isinstance(mv, dict):
                        d = dict(mv)
                        d.setdefault("key", mk)
                        tmp.append(d)
                    else:
                        tmp.append({"key": mk, "outcomes": mv})
                markets = tmp

            if not isinstance(markets, list):
                continue

            for m in markets:
                if not isinstance(m, dict):
                    continue
                market_type = _normalize_market_type(_infer_market_type(m))

                outcomes = m.get("outcomes") or m.get("selections") or m.get("prices") or []
                if isinstance(outcomes, dict):
                    out2 = []
                    for ok, ov in outcomes.items():
                        if isinstance(ov, dict):
                            d = dict(ov)
                            d.setdefault("name", ok)
                            out2.append(d)
                    outcomes = out2

                if not isinstance(outcomes, list):
                    continue

                for o in outcomes:
                    if not isinstance(o, dict):
                        continue
                    outcome_name = _infer_outcome_name(o)
                    if not outcome_name:
                        continue
                    dec = _infer_outcome_decimal(o)
                    if not dec or dec <= 1.0:
                        continue

                    line = _infer_outcome_line(o)

                    ok = _normalize_name(_guess_over_under_name(outcome_name))
                    oid = _make_outcome_id(ok, line)

                    rows.append(TapeRow(
                        event_id=str(event_id),
                        market_type=str(market_type),
                        line=line,
                        outcome_name=str(outcome_name),
                        outcome_key=ok,
                        outcome_id=oid,
                        book=str(book),
                        decimal=float(dec),
                    ))

    return rows


# -----------------------------
# No-vig + Consensus engine
# -----------------------------
def _group_market(rows: List[TapeRow], event_id: str, market_type: str) -> List[TapeRow]:
    eid = str(event_id)
    mkt = _normalize_market_type(str(market_type))
    return [r for r in rows if r.event_id == eid and r.market_type == mkt]


def _market_snapshot(rows: List[TapeRow], max_rows: int = 5000) -> dict:
    outs = []
    # cap snapshot size (debug)
    for r in rows[:max_rows]:
        outs.append({
            "book": r.book,
            "name": r.outcome_name,
            "name_key": r.outcome_key,
            "line": r.line,
            "outcome_id": r.outcome_id,
            "decimal": r.decimal,
            "implied": r.implied,
        })
    return {"outcomes": outs, "rows": len(rows)}


def _best_price_per_book_outcome(market_rows: List[TapeRow]) -> Dict[str, Dict[str, TapeRow]]:
    """
    For each book and outcome_id keep the best price (max decimal, min implied).
    """
    by_book: Dict[str, Dict[str, TapeRow]] = {}
    for r in market_rows:
        m = by_book.setdefault(r.book, {})
        prev = m.get(r.outcome_id)
        if prev is None or r.decimal > prev.decimal:
            m[r.outcome_id] = r
    return by_book


def _no_vig_by_book(
    market_rows: List[TapeRow],
    min_outcomes_per_book: int = 2,
    overround_min: float = 0.90,
    overround_max: float = 1.25,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, int]]:
    """
    Returns:
      fair_probs_by_book[book][outcome_id] = fair_prob (normalized)
      overround_by_book[book] = sum(implied)
      outcomes_by_book[book] = number of outcomes used
    """
    best = _best_price_per_book_outcome(market_rows)

    fair_by_book: Dict[str, Dict[str, float]] = {}
    overround: Dict[str, float] = {}
    outcomes_n: Dict[str, int] = {}

    for book, omap in best.items():
        if len(omap) < min_outcomes_per_book:
            continue

        imp_map: Dict[str, float] = {}
        for oid, row in omap.items():
            p = row.implied
            if p > 0:
                imp_map[oid] = p

        if len(imp_map) < min_outcomes_per_book:
            continue

        ov = sum(imp_map.values())
        if ov <= 0:
            continue

        # sanity filter to reject corrupted/partial markets
        if not (overround_min <= ov <= overround_max):
            continue

        overround[book] = ov
        outcomes_n[book] = len(imp_map)
        fair_by_book[book] = {oid: p / ov for oid, p in imp_map.items()}

    return fair_by_book, overround, outcomes_n


def _collect_outcomes(fair_by_book: Dict[str, Dict[str, float]]) -> List[str]:
    all_outcomes: set[str] = set()
    for mp in fair_by_book.values():
        all_outcomes |= set(mp.keys())
    return sorted(all_outcomes)


def _consensus_probs(
    fair_by_book: Dict[str, Dict[str, float]],
    method: str = "median",
    trim: float = 0.2,
    min_books: int = 2,
    book_weights: Optional[Dict[str, float]] = None,
    outlier_z: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, dict]]:
    """
    Compute consensus fair probs across books.

    Methods:
      - median
      - trimmed_mean
      - winsorized_mean
      - weighted_median   (uses book_weights)
    """
    book_weights = { _normalize_book(k): float(v) for k, v in (book_weights or {}).items() if v is not None and float(v) > 0 }
    books = list(fair_by_book.keys())

    consensus: Dict[str, float] = {}
    diag: Dict[str, dict] = {}

    outcomes = _collect_outcomes(fair_by_book)

    for oid in outcomes:
        vals: List[float] = []
        wts: List[float] = []
        used_books: List[str] = []

        for b in books:
            mp = fair_by_book.get(b, {})
            v = mp.get(oid)
            if v is None:
                continue
            v = float(v)
            if not (0.0 < v < 1.0):
                continue
            vals.append(v)
            wts.append(float(book_weights.get(b, 1.0)))
            used_books.append(b)

        if len(vals) < min_books:
            continue

        # optional outlier filtering (z-score)
        filtered = list(range(len(vals)))
        if outlier_z is not None and len(vals) >= 5:
            mu = sum(vals) / len(vals)
            var = sum((x - mu) ** 2 for x in vals) / max(1, (len(vals) - 1))
            sd = math.sqrt(var) if var > 0 else 0.0
            if sd > 0:
                keep = []
                for i, x in enumerate(vals):
                    z = abs((x - mu) / sd)
                    if z <= outlier_z:
                        keep.append(i)
                if len(keep) >= min_books:
                    filtered = keep

        vals_f = [vals[i] for i in filtered]
        wts_f = [wts[i] for i in filtered]
        used_f = [used_books[i] for i in filtered]

        if method == "median":
            c = float(median(vals_f))
        elif method == "trimmed_mean":
            c = float(_trimmed_mean(vals_f, trim))
        elif method == "winsorized_mean":
            c = float(_winsorized_mean(vals_f, trim))
        elif method == "weighted_median":
            c = float(_weighted_median(vals_f, wts_f))
        else:
            c = float(median(vals_f))

        if not (0.0 < c < 1.0):
            continue

        consensus[oid] = c
        diag[oid] = {
            "books": used_f,
            "n": len(vals_f),
            "min": min(vals_f),
            "median": float(median(vals_f)),
            "max": max(vals_f),
        }

    # Re-normalize across outcomes (important if some runners missing)
    s = sum(consensus.values())
    if s > 0:
        for k in list(consensus.keys()):
            consensus[k] = _clamp01(consensus[k] / s)

    return consensus, diag


def compute_market_consensus(
    tape: List[TapeRow],
    event_id: str,
    market_type: str,
    method: str,
    trim: float,
    min_books: int,
    min_outcomes_per_book: int,
    overround_min: float,
    overround_max: float,
    book_weights: Optional[Dict[str, float]],
    outlier_z: Optional[float],
) -> Tuple[dict, Optional[str]]:
    """
    Returns:
      market_bundle = {
        "event_id", "market_type",
        "snapshot": {...},
        "no_vig": {"consensus": {...}, "diag": {...}, "books_used": [...], "overround": {...}},
      }
    """
    mrows = _group_market(tape, event_id, market_type)
    snap = _market_snapshot(mrows)

    fair_by_book, overround, outcomes_n = _no_vig_by_book(
        mrows,
        min_outcomes_per_book=min_outcomes_per_book,
        overround_min=overround_min,
        overround_max=overround_max,
    )

    consensus, diag = _consensus_probs(
        fair_by_book=fair_by_book,
        method=method,
        trim=trim,
        min_books=min_books,
        book_weights=book_weights,
        outlier_z=outlier_z,
    )

    books_used = sorted(list(fair_by_book.keys()))

    if not consensus:
        return {
            "event_id": str(event_id),
            "market_type": _normalize_market_type(str(market_type)),
            "snapshot": snap,
            "no_vig": {
                "consensus": {},
                "diag": {},
                "books_used": books_used,
                "overround": overround,
                "outcomes_by_book": outcomes_n,
            },
        }, None

    return {
        "event_id": str(event_id),
        "market_type": _normalize_market_type(str(market_type)),
        "snapshot": snap,
        "no_vig": {
            "consensus": consensus,
            "diag": diag,
            "books_used": books_used,
            "overround": overround,
            "outcomes_by_book": outcomes_n,
        },
    }, "no_vig_consensus_v3"


def _pick_outcome_id(pick: dict) -> str:
    name = _pick_outcome_name(pick)
    line = _pick_line(pick)
    ok = _normalize_name(_guess_over_under_name(name))
    return _make_outcome_id(ok, line)


def p_market_for_pick(
    pick: dict,
    market_bundle: dict,
    fallback_to_pick_odds: bool = True,
) -> Tuple[Optional[float], Optional[str], int]:
    """
    Returns (p_market, method, books_used_count)
    """
    oid = _pick_outcome_id(pick)

    nv = (market_bundle.get("no_vig") or {})
    cons = (nv.get("consensus") or {})
    books_used = nv.get("books_used") or []

    if oid in cons and 0.0 < float(cons[oid]) < 1.0:
        return float(cons[oid]), "no_vig_consensus_v3", len(books_used)

    # fallback: if line mismatch, try name-only match (rare but useful when pick has no line)
    # if pick line is None, this will match the only runner by name if present.
    if "::" in oid and oid.endswith("::"):
        # line-less pick, try any consensus key that starts with "name::"
        prefix = oid
        for k, v in cons.items():
            if k.startswith(prefix) and 0.0 < float(v) < 1.0:
                return float(v), "no_vig_consensus_name_only", len(books_used)

    if fallback_to_pick_odds:
        amer = _coerce_american(pick.get("odds"))
        if amer is not None:
            p = _dec_to_imp(_amer_to_dec(amer))
            if 0.0 < p < 1.0:
                return float(p), "implied_from_pick_odds", len(books_used)

    return None, None, len(books_used)


# -----------------------------
# Runner (supports multiple gp APIs)
# -----------------------------

def _coerce_picks_output(out: Any) -> Optional[List[dict]]:
    # Accept:
    #   - list[dict]
    #   - dict with 'picks' key
    #   - tuple/list where first element is list[dict] or dict with 'picks'
    if out is None:
        return None

    if isinstance(out, list):
        # Could be picks list, or could be (picks, meta) packed weirdly (rare)
        # If it's a list of dicts, treat it as picks.
        if (not out) or all(isinstance(x, dict) for x in out):
            return out
        # Otherwise, fall through.
    if isinstance(out, dict):
        v = out.get("picks") or out.get("results") or out.get("data")
        if isinstance(v, list) and ((not v) or all(isinstance(x, dict) for x in v)):
            return v
    if isinstance(out, tuple) or (isinstance(out, list) and len(out) >= 1):
        first = out[0]
        if isinstance(first, list) and ((not first) or all(isinstance(x, dict) for x in first)):
            return first
        if isinstance(first, dict):
            v = first.get("picks") or first.get("results") or first.get("data")
            if isinstance(v, list) and ((not v) or all(isinstance(x, dict) for x in v)):
                return v

    return None

def _call_pick_generator(odds_blob: Any) -> List[dict]:
    # DEBUG_CALLS: instrument generator outputs

    """
    We try, in order:
      gp.generate_picks_full(odds_blob)
      gp.generate_picks_from_odds(odds_blob)
      gp.generate_picks(odds_blob)
      gp.generate_picks()
    """
    candidates = [
        "generate_picks_full",
        "generate_picks_from_odds",
        "generate_picks",
    ]

    for name in candidates:
        fn = getattr(gp, name, None)
        if not callable(fn):
            continue
        # Try with odds_blob first, then no-arg. Capture exceptions.
        attempts = []
        try:
            sig = inspect.signature(fn)
            attempts.append(("with_odds" if len(sig.parameters) >= 1 else "no_arg_by_sig", len(sig.parameters) >= 1))
        except Exception:
            # unknown signature; still try both
            attempts.append(("with_odds", True))
            attempts.append(("no_arg", False))

        # Ensure both modes are tried at least once
        if not any(a[1] for a in attempts):
            attempts.insert(0, ("with_odds", True))
        if not any((not a[1]) for a in attempts):
            attempts.append(("no_arg", False))

        last_err = None
        for tag, use_odds in attempts:
            try:
                out = fn(odds_blob) if use_odds else fn()
                picks = _coerce_picks_output(out)
                if picks is not None:
                    return picks
            except Exception as e:
                last_err = (tag, repr(e))
                continue
        # If callable but nothing usable returned, continue searching candidates.
        continue

    if callable(getattr(gp, "generate_picks", None)):
        out = gp.generate_picks()
    picks = _coerce_picks_output(out)
    if picks is not None:
        return picks
    raise SystemExit("generate_picks() returned an unsupported shape (expected list or dict with picks)")

    raise SystemExit("No compatible pick generator found in generate_picks.py (expected generate_picks[_full|_from_odds] to return list or dict with picks).")


def _dump_shape(x: Any, max_keys: int = 40) -> dict:
    """
    Debug tool: helps us align adapters without printing your entire odds blob.
    """
    out: Dict[str, Any] = {"type": type(x).__name__}
    if isinstance(x, dict):
        ks = list(x.keys())
        out["keys"] = ks[:max_keys]
        for k in ks[:10]:
            out[f"key:{k}:type"] = type(x.get(k)).__name__
    elif isinstance(x, list):
        out["len"] = len(x)
        if x:
            out["item0_type"] = type(x[0]).__name__
            if isinstance(x[0], dict):
                out["item0_keys"] = list(x[0].keys())[:max_keys]
    else:
        out["repr"] = repr(x)[:240]
    return out


def _load_json_file(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json_file(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_book_weights_json(s: str) -> Dict[str, float]:
    """
    Accepts:
      - raw JSON string: '{"draftkings":1.2,"fanduel":1.1}'
      - @file.json to load from file
    """
    s = (s or "").strip()
    if not s:
        return {}
    if s.startswith("@"):
        fp = Path(s[1:])
        obj = _load_json_file(fp)
        if isinstance(obj, dict):
            return { _normalize_book(k): float(v) for k, v in obj.items() if v is not None and float(v) > 0 }
        return {}
    obj = json.loads(s)
    if isinstance(obj, dict):
        return { _normalize_book(k): float(v) for k, v in obj.items() if v is not None and float(v) > 0 }
    return {}



def _env_snapshot() -> dict:
    # Snapshot common knobs that often zero-out pick generation.
    keys = [
        "SPORT", "SPORT_KEY", "LEAGUE", "REGION", "MARKETS", "BOOKS",
        "ODDS_API_KEY", "ODDSAPI_KEY",
        "MIN_EDGE_PCT", "MIN_EV", "MAX_DAILY_FRAC", "MAX_PER_GAME_FRAC",
        "BANKROLL", "DEBUG", "DEBUG_ALL",
        "COMMENCE_HOURS", "DAYS_FROM", "DATE_FROM", "DATE_TO",
    ]
    out = {}
    for k in keys:
        v = os.environ.get(k)
        if v is None:
            continue
        # don't leak full API key
        if "KEY" in k and v:
            out[k] = v[:4] + "…" + v[-2:] if len(v) > 8 else "…"
        else:
            out[k] = v
    return out

def _host_facts() -> dict:
    try:
        hn = socket.gethostname()
    except Exception:
        hn = "unknown"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": hn,
        "pid": os.getpid(),
    }


def _git_head_short() -> Optional[str]:
    # Avoid calling git; use common env vars if present
    for k in ("GIT_COMMIT", "COMMIT_SHA", "SOURCE_VERSION"):
        v = os.environ.get(k, "").strip()
        if v:
            return v[:12]
    return None


# -----------------------------
# Main
# -----------------------------

def _normalize_market_name(x: str) -> str:
    x = (x or '').strip().lower()
    if x in ('moneyline','ml','h2h'):
        return 'moneyline'
    if x in ('spread','spreads','handicap'):
        return 'spread'
    if x in ('total','totals','over_under','ou'):
        return 'total'
    return x or 'unknown'


def _normalize_pick_schema(pk: dict) -> dict:
    # Upgrade any legacy/model pick into the canonical schema expected by the app.
    # Output keys: event_id, market, selection, book, odds, decimal, line, implied, p_model, p_market,
    #              edge, edge_pct, ev_per_1, stake, meta
    if not isinstance(pk, dict):
        return {'event_id': '', 'market': 'unknown', 'selection': str(pk), 'book': 'unknown', 'meta': {}}

    out = dict(pk)

    # market
    market = out.get('market') or out.get('market_type')
    if not market:
        market = out.get('bet_type') or out.get('type')
    out['market'] = _normalize_market_name(str(market or 'unknown'))

    # selection
    sel = out.get('selection') or out.get('name') or out.get('team') or out.get('outcome')
    out['selection'] = str(sel or '')

    # book
    out['book'] = str(out.get('book') or out.get('sportsbook') or out.get('site') or 'unknown')

    # odds/decimal
    amer = out.get('odds')
    dec = out.get('decimal')
    if dec is None and amer is not None:
        try:
            a = int(float(str(amer).strip()))
            dec = _amer_to_dec(a)
        except Exception:
            dec = None
    if amer is None and dec is not None:
        try:
            amer = _dec_to_american(float(dec)) if ' _dec_to_american' in globals() else None
        except Exception:
            amer = None
    if dec is not None:
        try:
            out['decimal'] = float(dec)
        except Exception:
            pass
    if amer is not None:
        try:
            out['odds'] = int(float(amer))
        except Exception:
            out['odds'] = amer

    # line
    if 'line' not in out:
        out['line'] = out.get('points') or out.get('handicap') or out.get('total') or None

    # implied
    if 'implied' not in out:
        if 'implied_prob' in out:
            try:
                out['implied'] = float(out['implied_prob'])
            except Exception:
                pass
        elif dec is not None:
            try:
                out['implied'] = _dec_to_imp(float(dec))
            except Exception:
                pass

    # p_model
    if 'p_model' not in out:
        if 'model_prob' in out:
            try:
                out['p_model'] = float(out['model_prob'])
            except Exception:
                pass
        else:
            pm = _ensure_model_prob(out) if '_ensure_model_prob' in globals() else None
            if pm is not None:
                out['p_model'] = float(pm)

    # p_market (best effort)
    if 'p_market' not in out:
        for k in ('market_prob','p_mkt','consensus_prob'):
            if k in out:
                try:
                    out['p_market'] = float(out[k])
                    break
                except Exception:
                    pass

    # edge/edge_pct
    if 'edge_pct' in out and 'edge' not in out:
        try:
            out['edge'] = float(out['edge_pct']) / 100.0
        except Exception:
            pass
    if 'edge' in out and 'edge_pct' not in out:
        try:
            out['edge_pct'] = float(out['edge']) * 100.0
        except Exception:
            pass

    # EV
    if 'ev_per_1' not in out and 'ev' in out:
        try:
            out['ev_per_1'] = float(out['ev'])
        except Exception:
            pass

    # meta
    meta = out.get('meta')
    if not isinstance(meta, dict):
        meta = {}
    if 'matchup' in out and 'matchup' not in meta:
        meta['matchup'] = out.get('matchup')
    out['meta'] = meta

    # event_id
    if 'event_id' not in out:
        out['event_id'] = str(out.get('eventId') or out.get('id') or out.get('game_id') or '')

    return out


def _normalize_all_picks(picks: list) -> list:
    if not isinstance(picks, list):
        return []
    return [_normalize_pick_schema(x) for x in picks if x is not None]


def main() -> int:
    ap = argparse.ArgumentParser(description="Covariant full ledger generator (trader-grade market consensus v3).")
    ap.add_argument("--out", default="", help="Output file")
    ap.add_argument("--consensus", default="weighted_median",
                    choices=["median","trimmed_mean","winsorized_mean","weighted_median"])
    ap.add_argument("--trim", type=float, default=0.2)
    ap.add_argument("--min-books", type=int, default=2)
    ap.add_argument("--min-outcomes-per-book", type=int, default=2)
    ap.add_argument("--overround-min", type=float, default=0.90)
    ap.add_argument("--overround-max", type=float, default=1.25)
    ap.add_argument("--outlier-z", type=float, default=0.0)
    ap.add_argument("--dump-odds-shape", action="store_true")

    args = ap.parse_args()

    # -------------------------
    # Fetch odds + build tape
    # -------------------------
    odds_blob = gp.fetch_odds()
    tape = _extract_market_rows(odds_blob)

    # -------------------------
    # Try model generator first
    # -------------------------
    picks = _call_pick_generator(odds_blob)

    # -------------------------
    # FALLBACK: market-only picks
    # -------------------------
    if not picks:
        print("⚠️ Model produced 0 picks — using market-only fallback.")

        bankroll = float(os.environ.get("BANKROLL","500"))
        max_daily_frac = float(os.environ.get("MAX_DAILY_FRAC","0.10"))
        max_per_game_frac = float(os.environ.get("MAX_PER_GAME_FRAC","0.04"))

        horizon_hours = float(os.environ.get("HORIZON_HOURS","36"))
        kelly_mult = float(os.environ.get("KELLY_MULT","0.25"))
        top_n = int(os.environ.get("TOP_N","25"))

        picks = _fallback_generate_value_picks(
            odds_blob=odds_blob,
            tape=tape,
            consensus_method=args.consensus,
            trim=args.trim,
            min_books=args.min_books,
            min_outcomes_per_book=args.min_outcomes_per_book,
            overround_min=args.overround_min,
            overround_max=args.overround_max,
            book_weights=None,
            outlier_z=args.outlier_z if args.outlier_z > 0 else None,
            horizon_hours=horizon_hours,
            bankroll=bankroll,
            max_daily_frac=max_daily_frac,
            max_per_game_frac=max_per_game_frac,
            kelly_mult=kelly_mult,
            top_n=top_n,
        )

        print(f"✅ Fallback created {len(picks)} picks")

    # -------------------------
    # Save output
    # -------------------------
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outp = Path(args.out) if args.out else (RUNS_DIR / f"picks_full_{stamp}.json")

    picks = _normalize_all_picks(picks)

    payload = {
        "timestamp": int(dt.datetime.now().timestamp()),
        "meta": {
            "generated_at": _now_iso(),
            "engine": "fallback_consensus_v3",
            "tape_rows": len(tape),
            "picks": len(picks),
        },
        "picks": picks,
    }

    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"(saved) {outp}")
    print(f"coverage: picks={len(picks)} | tape_rows={len(tape)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

    raise SystemExit(main())
