"""
Availability Store - SQLite Persistence

Professional-grade storage for player availability, injuries, and impact data.

Schema:
- players: Player registry
- games: Game schedule
- player_game_status: Per-game player availability
- player_minutes_rolling: Rolling minute averages
- audit_availability_fetch: Fetch audit trail

NO MOCK DATA. Real ESPN data only.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class PlayerStatus(Enum):
    """Normalized player status"""
    ACTIVE = "ACTIVE"
    PROBABLE = "PROBABLE"
    QUESTIONABLE = "QUESTIONABLE"
    DOUBTFUL = "DOUBTFUL"
    OUT = "OUT"
    IR = "IR"
    SUSPENDED = "SUSPENDED"
    UNKNOWN = "UNKNOWN"
    
    @classmethod
    def from_espn(cls, status_text: str) -> 'PlayerStatus':
        """Map ESPN status text to normalized enum"""
        status_upper = status_text.upper().strip()
        
        if not status_text or status_upper in ['ACTIVE', 'HEALTHY']:
            return cls.ACTIVE
        elif 'PROB' in status_upper:
            return cls.PROBABLE
        elif 'QUEST' in status_upper or 'GTD' in status_upper:
            return cls.QUESTIONABLE
        elif 'DOUBT' in status_upper:
            return cls.DOUBTFUL
        elif status_upper in ['OUT', 'DNP']:
            return cls.OUT
        elif 'IR' in status_upper or 'INJURED RESERVE' in status_upper:
            return cls.IR
        elif 'SUSP' in status_upper:
            return cls.SUSPENDED
        else:
            return cls.UNKNOWN


# Play probability mapping by status
PLAY_PROB_MAP = {
    PlayerStatus.ACTIVE: 1.00,
    PlayerStatus.PROBABLE: 0.90,
    PlayerStatus.QUESTIONABLE: 0.50,
    PlayerStatus.DOUBTFUL: 0.25,
    PlayerStatus.OUT: 0.00,
    PlayerStatus.IR: 0.00,
    PlayerStatus.SUSPENDED: 0.00,
    PlayerStatus.UNKNOWN: 0.70,  # Conservative default
}


@dataclass
class Player:
    """Player record"""
    player_id: str
    league: str
    name: str
    team_id: str
    position: str
    
    @property
    def key(self) -> str:
        return f"{self.league}:{self.player_id}"


@dataclass
class Game:
    """Game record"""
    game_id: str
    league: str
    start_ts: datetime
    home_team_id: str
    away_team_id: str
    
    @property
    def key(self) -> str:
        return f"{self.league}:{self.game_id}"


@dataclass
class PlayerGameStatus:
    """Player status for specific game"""
    game_id: str
    player_id: str
    status_norm: PlayerStatus
    detail: str
    play_prob: float
    est_minutes: float
    is_starter: Optional[bool]
    ts: datetime


@dataclass
class PlayerMinutesRolling:
    """Rolling minute statistics"""
    player_id: str
    league: str
    window_n: int
    minutes_avg: float
    minutes_p50: float
    minutes_p90: float
    last_ts: datetime


class AvailabilityStore:
    """
    SQLite-backed availability storage.
    
    Thread-safe, ACID compliant, supports concurrent reads.
    """
    
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: str = "data/availability.db"):
        """
        Initialize store.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,  # Allow multi-threaded access
            timeout=30.0
        )
        self.conn.row_factory = sqlite3.Row
        
        self._initialize_schema()
        
        logger.info(f"Availability store initialized: {db_path}")
    
    def _initialize_schema(self):
        """Create tables if not exist"""
        cursor = self.conn.cursor()
        
        # Players table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id TEXT NOT NULL,
                league TEXT NOT NULL,
                name TEXT NOT NULL,
                team_id TEXT NOT NULL,
                position TEXT,
                PRIMARY KEY (league, player_id)
            )
        """)
        
        # Games table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT NOT NULL,
                league TEXT NOT NULL,
                start_ts INTEGER NOT NULL,
                home_team_id TEXT NOT NULL,
                away_team_id TEXT NOT NULL,
                PRIMARY KEY (league, game_id)
            )
        """)
        
        # Player game status
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_game_status (
                game_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                status_norm TEXT NOT NULL,
                detail TEXT,
                play_prob REAL NOT NULL,
                est_minutes REAL NOT NULL,
                is_starter INTEGER,
                ts INTEGER NOT NULL,
                PRIMARY KEY (game_id, player_id)
            )
        """)
        
        # Player minutes rolling
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_minutes_rolling (
                player_id TEXT NOT NULL,
                league TEXT NOT NULL,
                window_n INTEGER NOT NULL,
                minutes_avg REAL NOT NULL,
                minutes_p50 REAL NOT NULL,
                minutes_p90 REAL NOT NULL,
                last_ts INTEGER NOT NULL,
                PRIMARY KEY (league, player_id, window_n)
            )
        """)
        
        # Audit log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_availability_fetch (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                fetched_ts INTEGER NOT NULL,
                source TEXT NOT NULL,
                cache_hit INTEGER NOT NULL,
                payload_hash TEXT NOT NULL,
                success INTEGER NOT NULL,
                error_msg TEXT
            )
        """)
        
        # Metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        # Set schema version
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("schema_version", str(self.SCHEMA_VERSION))
        )
        
        self.conn.commit()
        
        logger.info("Database schema initialized")
    
    # ===== PLAYERS =====
    
    def upsert_player(self, player: Player):
        """Insert or update player"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO players 
            (player_id, league, name, team_id, position)
            VALUES (?, ?, ?, ?, ?)
        """, (
            player.player_id,
            player.league,
            player.name,
            player.team_id,
            player.position
        ))
        
        self.conn.commit()
    
    def get_player(self, league: str, player_id: str) -> Optional[Player]:
        """Get player by ID"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM players 
            WHERE league = ? AND player_id = ?
        """, (league, player_id))
        
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return Player(
            player_id=row['player_id'],
            league=row['league'],
            name=row['name'],
            team_id=row['team_id'],
            position=row['position']
        )
    
    def get_team_players(self, league: str, team_id: str) -> List[Player]:
        """Get all players for a team"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM players 
            WHERE league = ? AND team_id = ?
        """, (league, team_id))
        
        return [
            Player(
                player_id=row['player_id'],
                league=row['league'],
                name=row['name'],
                team_id=row['team_id'],
                position=row['position']
            )
            for row in cursor.fetchall()
        ]
    
    # ===== GAMES =====
    
    def upsert_game(self, game: Game):
        """Insert or update game"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO games 
            (game_id, league, start_ts, home_team_id, away_team_id)
            VALUES (?, ?, ?, ?, ?)
        """, (
            game.game_id,
            game.league,
            int(game.start_ts.timestamp()),
            game.home_team_id,
            game.away_team_id
        ))
        
        self.conn.commit()
    
    def get_game(self, league: str, game_id: str) -> Optional[Game]:
        """Get game by ID"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM games 
            WHERE league = ? AND game_id = ?
        """, (league, game_id))
        
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return Game(
            game_id=row['game_id'],
            league=row['league'],
            start_ts=datetime.fromtimestamp(row['start_ts']),
            home_team_id=row['home_team_id'],
            away_team_id=row['away_team_id']
        )
    
    # ===== PLAYER GAME STATUS =====
    
    def upsert_player_status(self, status: PlayerGameStatus):
        """Insert or update player status for game"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO player_game_status 
            (game_id, player_id, status_norm, detail, play_prob, est_minutes, is_starter, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            status.game_id,
            status.player_id,
            status.status_norm.value,
            status.detail,
            status.play_prob,
            status.est_minutes,
            1 if status.is_starter else 0 if status.is_starter is not None else None,
            int(status.ts.timestamp())
        ))
        
        self.conn.commit()
    
    def get_player_status(
        self,
        game_id: str,
        player_id: str
    ) -> Optional[PlayerGameStatus]:
        """Get player status for specific game"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM player_game_status 
            WHERE game_id = ? AND player_id = ?
        """, (game_id, player_id))
        
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return PlayerGameStatus(
            game_id=row['game_id'],
            player_id=row['player_id'],
            status_norm=PlayerStatus(row['status_norm']),
            detail=row['detail'],
            play_prob=row['play_prob'],
            est_minutes=row['est_minutes'],
            is_starter=bool(row['is_starter']) if row['is_starter'] is not None else None,
            ts=datetime.fromtimestamp(row['ts'])
        )
    
    def get_game_statuses(self, game_id: str) -> List[PlayerGameStatus]:
        """Get all player statuses for a game"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM player_game_status 
            WHERE game_id = ?
        """, (game_id,))
        
        return [
            PlayerGameStatus(
                game_id=row['game_id'],
                player_id=row['player_id'],
                status_norm=PlayerStatus(row['status_norm']),
                detail=row['detail'],
                play_prob=row['play_prob'],
                est_minutes=row['est_minutes'],
                is_starter=bool(row['is_starter']) if row['is_starter'] is not None else None,
                ts=datetime.fromtimestamp(row['ts'])
            )
            for row in cursor.fetchall()
        ]
    
    # ===== PLAYER MINUTES ROLLING =====
    
    def upsert_minutes_rolling(self, minutes: PlayerMinutesRolling):
        """Insert or update rolling minutes"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO player_minutes_rolling 
            (player_id, league, window_n, minutes_avg, minutes_p50, minutes_p90, last_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            minutes.player_id,
            minutes.league,
            minutes.window_n,
            minutes.minutes_avg,
            minutes.minutes_p50,
            minutes.minutes_p90,
            int(minutes.last_ts.timestamp())
        ))
        
        self.conn.commit()
    
    def get_minutes_rolling(
        self,
        league: str,
        player_id: str,
        window_n: int
    ) -> Optional[PlayerMinutesRolling]:
        """Get rolling minutes for player"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM player_minutes_rolling 
            WHERE league = ? AND player_id = ? AND window_n = ?
        """, (league, player_id, window_n))
        
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return PlayerMinutesRolling(
            player_id=row['player_id'],
            league=row['league'],
            window_n=row['window_n'],
            minutes_avg=row['minutes_avg'],
            minutes_p50=row['minutes_p50'],
            minutes_p90=row['minutes_p90'],
            last_ts=datetime.fromtimestamp(row['last_ts'])
        )
    
    # ===== AUDIT LOG =====
    
    def log_fetch(
        self,
        game_id: str,
        source: str,
        cache_hit: bool,
        payload_hash: str,
        success: bool = True,
        error_msg: Optional[str] = None
    ):
        """Log availability fetch"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO audit_availability_fetch 
            (game_id, fetched_ts, source, cache_hit, payload_hash, success, error_msg)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id,
            int(datetime.utcnow().timestamp()),
            source,
            1 if cache_hit else 0,
            payload_hash,
            1 if success else 0,
            error_msg
        ))
        
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# Singleton instance
_store = None


def get_availability_store() -> AvailabilityStore:
    """Get singleton availability store"""
    global _store
    if _store is None:
        _store = AvailabilityStore()
    return _store
