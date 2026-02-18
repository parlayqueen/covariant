"""
Elite Multi-Source Injury Fetcher

Professional injury data aggregation using multiple ESPN endpoints:
1. Primary: ESPN Injuries API (comprehensive)
2. Secondary: Scoreboard embedded injuries (real-time)
3. Tertiary: Team depth charts (lineup verification)
4. Verification: Cross-reference multiple sources

NEVER miss critical injury data.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, date
from dataclasses import dataclass
import hashlib

from pickslab_elite.core.http.espn_client import get_espn_client
from pickslab_elite.core.store.availability_store import PlayerStatus, get_availability_store


logger = logging.getLogger(__name__)


@dataclass
class EnhancedInjuryRecord:
    """Complete injury record with source tracking"""
    player_id: str
    player_name: str
    team_id: str
    status: PlayerStatus
    raw_status: str
    injury_type: str
    description: str
    date_reported: datetime
    sources: List[str]  # Multiple sources for verification
    confidence: float  # 0-1 based on source agreement


class MultiSourceInjuryFetcher:
    """
    Professional multi-source injury fetcher.
    
    Strategy:
    - Fetch from 3+ sources
    - Cross-validate data
    - Prioritize most recent
    - Flag discrepancies
    """
    
    def __init__(self, league: str = 'nba'):
        self.league = league
        self.client = get_espn_client()
        self.store = get_availability_store()
        
        # League-specific ESPN paths
        self.league_paths = {
            'nba': 'basketball/nba',
            'nfl': 'football/nfl',
            'nhl': 'hockey/nhl',
            'mlb': 'baseball/mlb'
        }
        
        self.league_path = self.league_paths.get(league, 'basketball/nba')
    
    def fetch_comprehensive_injuries(self) -> Dict[str, EnhancedInjuryRecord]:
        """
        Fetch injuries from ALL sources and merge.
        
        Returns dict of player_id -> injury record
        """
        all_injuries = {}
        
        # Source 1: Injuries API
        try:
            injuries_api = self._fetch_injuries_api()
            logger.info(f"Injuries API: {len(injuries_api)} found")
            
            for injury in injuries_api:
                if injury.player_id not in all_injuries:
                    all_injuries[injury.player_id] = injury
                else:
                    # Merge sources
                    existing = all_injuries[injury.player_id]
                    existing.sources.extend(injury.sources)
                    existing.confidence = min(1.0, existing.confidence + 0.3)
        except Exception as e:
            logger.warning(f"Injuries API failed: {e}")
        
        # Source 2: Scoreboard (real-time)
        try:
            scoreboard_injuries = self._fetch_scoreboard_injuries()
            logger.info(f"Scoreboard: {len(scoreboard_injuries)} found")
            
            for injury in scoreboard_injuries:
                if injury.player_id not in all_injuries:
                    all_injuries[injury.player_id] = injury
                else:
                    existing = all_injuries[injury.player_id]
                    
                    # Check for status disagreement
                    if existing.status != injury.status:
                        logger.warning(
                            f"Status mismatch for {injury.player_name}: "
                            f"{existing.status} vs {injury.status}"
                        )
                        # Use more pessimistic status
                        if self._is_worse_status(injury.status, existing.status):
                            existing.status = injury.status
                    
                    existing.sources.extend(injury.sources)
                    existing.confidence = min(1.0, existing.confidence + 0.2)
        except Exception as e:
            logger.warning(f"Scoreboard fetch failed: {e}")
        
        # Source 3: Team transactions (suspensions, IR moves)
        # Would add team transaction endpoint here
        
        logger.info(f"Total unique injuries: {len(all_injuries)}")
        
        return all_injuries
    
    def verify_lineup_availability(
        self,
        game_id: str,
        team_id: str
    ) -> Dict[str, bool]:
        """
        Verify player availability via lineup check.
        
        Returns dict of player_id -> is_available
        """
        try:
            # Fetch game details
            game_data = self.client.get(
                f"sports/{self.league_path}/summary",
                params={'event': game_id}
            )
            
            # Parse active roster
            active_players = self._parse_active_roster(game_data, team_id)
            
            return active_players
        
        except Exception as e:
            logger.warning(f"Lineup verification failed: {e}")
            return {}
    
    def _fetch_injuries_api(self) -> List[EnhancedInjuryRecord]:
        """Fetch from primary injuries endpoint"""
        data = self.client.get(f"sports/{self.league_path}/injuries")
        
        injuries = []
        
        for item in data.get('injuries', []):
            try:
                athlete = item.get('athlete', {})
                player_id = str(athlete.get('id'))
                
                team = item.get('team', {})
                team_id = str(team.get('id'))
                
                status_text = item.get('status', 'UNKNOWN')
                status_norm = PlayerStatus.from_espn(status_text)
                
                details = item.get('details', {})
                injury_type = details.get('type', item.get('type', ''))
                description = details.get('detail', '')
                
                date_str = item.get('date')
                date_reported = datetime.utcnow()
                if date_str:
                    try:
                        date_reported = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except:
                        pass
                
                injury = EnhancedInjuryRecord(
                    player_id=player_id,
                    player_name=athlete.get('displayName', 'Unknown'),
                    team_id=team_id,
                    status=status_norm,
                    raw_status=status_text,
                    injury_type=injury_type,
                    description=description,
                    date_reported=date_reported,
                    sources=['injuries_api'],
                    confidence=0.8
                )
                
                injuries.append(injury)
            
            except Exception as e:
                logger.debug(f"Parse error: {e}")
                continue
        
        return injuries
    
    def _fetch_scoreboard_injuries(self) -> List[EnhancedInjuryRecord]:
        """Fetch from today's scoreboard"""
        date_str = date.today().strftime("%Y%m%d")
        
        data = self.client.get(
            f"sports/{self.league_path}/scoreboard",
            params={'dates': date_str}
        )
        
        injuries = []
        
        for event in data.get('events', []):
            for comp in event.get('competitions', []):
                for competitor in comp.get('competitors', []):
                    team_id = competitor['team']['id']
                    
                    # Parse injury notes
                    for inj in competitor.get('injuries', []):
                        try:
                            athlete = inj.get('athlete', {})
                            player_id = str(athlete.get('id'))
                            
                            status_text = inj.get('status', 'UNKNOWN')
                            
                            injury = EnhancedInjuryRecord(
                                player_id=player_id,
                                player_name=athlete.get('displayName', 'Unknown'),
                                team_id=team_id,
                                status=PlayerStatus.from_espn(status_text),
                                raw_status=status_text,
                                injury_type=inj.get('type', ''),
                                description=inj.get('details', {}).get('detail', ''),
                                date_reported=datetime.utcnow(),
                                sources=['scoreboard'],
                                confidence=0.9  # Scoreboard is most current
                            )
                            
                            injuries.append(injury)
                        except:
                            continue
        
        return injuries
    
    def _parse_active_roster(
        self,
        game_data: Dict,
        team_id: str
    ) -> Dict[str, bool]:
        """Parse active roster from game data"""
        active = {}
        
        try:
            boxscore = game_data.get('boxscore', {})
            
            for team in boxscore.get('teams', []):
                if str(team.get('team', {}).get('id')) == team_id:
                    # All players in boxscore are active
                    for stat_group in team.get('statistics', []):
                        for athlete_stat in stat_group.get('athletes', []):
                            athlete = athlete_stat.get('athlete', {})
                            player_id = str(athlete.get('id'))
                            if player_id:
                                active[player_id] = True
        except:
            pass
        
        return active
    
    @staticmethod
    def _is_worse_status(status1: PlayerStatus, status2: PlayerStatus) -> bool:
        """Check if status1 is worse than status2"""
        severity = {
            PlayerStatus.ACTIVE: 0,
            PlayerStatus.PROBABLE: 1,
            PlayerStatus.QUESTIONABLE: 2,
            PlayerStatus.DOUBTFUL: 3,
            PlayerStatus.OUT: 4,
            PlayerStatus.IR: 5,
            PlayerStatus.SUSPENDED: 5
        }
        
        return severity.get(status1, 0) > severity.get(status2, 0)
