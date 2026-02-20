#!/usr/bin/env python3
import os
import time
import json
import hashlib
import requests

LEAGUE_PATHS = {
    "nfl": ("football", "nfl"),
    "nba": ("basketball", "nba"),
    "mlb": ("baseball", "mlb"),
    "nhl": ("hockey", "nhl"),
}

class ESPNClient:
    def __init__(self, league="nfl", cache_dir=".cache/espn", ttl=60):
        if league not in LEAGUE_PATHS:
            raise ValueError("Unsupported league")
        self.league = league
        self.cache_dir = cache_dir
        self.ttl = ttl
        os.makedirs(cache_dir, exist_ok=True)

        sport, lg = LEAGUE_PATHS[league]
        self.scoreboard_url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{lg}/scoreboard"
        self.summary_url = f"https://site.web.api.espn.com/apis/site/v2/sports/{sport}/{lg}/summary"

        self.session = requests.Session()
        self.last_call = 0

    # ---------- caching ----------
    def _cache_key(self, url, params):
        raw = json.dumps({"u":url,"p":params},sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def _cache_path(self, key):
        return os.path.join(self.cache_dir, key + ".json")

    def _get_cache(self, key):
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        if time.time() - os.path.getmtime(path) > self.ttl:
            return None
        with open(path,"r") as f:
            return json.load(f)

    def _save_cache(self, key, data):
        with open(self._cache_path(key),"w") as f:
            json.dump(data,f)

    # ---------- http ----------
    def get_json(self, url, params=None):
        key = self._cache_key(url, params)
        cached = self._get_cache(key)
        if cached:
            return cached

        if time.time() - self.last_call < 0.35:
            time.sleep(0.35)

        r = self.session.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        self._save_cache(key, data)
        self.last_call = time.time()
        return data

    # ---------- endpoints ----------
    def scoreboard(self, dates=None):
        params = {}
        if dates:
            params["dates"] = dates
        return self.get_json(self.scoreboard_url, params)

    def summary(self, event_id):
        return self.get_json(self.summary_url, {"event": event_id})

    # ---------- parsers ----------
    @staticmethod
    def extract_matchups(payload):
        rows = []
        for ev in payload.get("events", []):
            comp = ev["competitions"][0]
            teams = comp["competitors"]

            home = next(t for t in teams if t["homeAway"]=="home")
            away = next(t for t in teams if t["homeAway"]=="away")

            rows.append({
                "event_id": ev["id"],
                "date": ev["date"],
                "home": home["team"]["abbreviation"],
                "away": away["team"]["abbreviation"],
                "home_score": home.get("score"),
                "away_score": away.get("score"),
            })
        return rows
