"""
ESPN NBA Adapter

Real ESPN API integration for NBA data:
- Scoreboard (games, teams)
- Rosters (players)
- Injuries/Status
- Game details

NO MOCK DATA. Real ESPN site.api endpoints only.
"""

import logging
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date

from pickslab_elite.core.http.espn_client import get_espn_client, ESPNClientError
from pickslab_elite.core.store.availability_store import (
    get_availability_store,
    Player, Game, PlayerGameStatus, PlayerStatus,
    PLAY_PROB_MAP
)


logger = logging.getLogger(__name__)


class ESPNNBAAdapter:
    """
    NBA data adapter using real ESPN endpoints.
    
    Endpoints:
    - /sports/basketball/nba/scoreboard
    - /sports/basketball/nba/teams/{team_id}/roster
    - /sports/basketball/nba/teams/{team_id}
    """
    
    LEAGUE = "nba"
    
    def __init__(self):
        self.client = get_espn_client()
        self.store = get_availability_store()
    
    def update_availability_for_date(self, target_date: date) -> Dict:
        """
        Update all availability data for a specific date.
        
        Args:
            target_date: Date to fetch (YYYY-MM-DD)
        
        Returns:
            Summary stats
        """
        logger.info(f"Updating NBA availability for {target_date}")
        
        stats = {
            'games_found': 0,
            'players_updated': 0,
            'statuses_updated': 0,
            'errors': []
        }
        
        try:
            # Fetch scoreboard for date
            scoreboard = self._fetch_scoreboard(target_date)
            
            # Extract games
            games = self._parse_games(scoreboard)
            stats['games_found'] = len(games)
            
            # Process each game
            for game in games:
                try:
                    # Store game
                    self.store.upsert_game(game)
                    
                    # Fetch rosters for both teams
                    home_players = self._fetch_team_roster(game.home_team_id)
                    away_players = self._fetch_team_roster(game.away_team_id)
                    
                    all_players = home_players + away_players
                    
                    # Store players
                    for player in all_players:
                        self.store.upsert_player(player)
                        stats['players_updated'] += 1
                    
                    # Fetch injury/status data
                    statuses = self._fetch_game_statuses(game, all_players)
                    
                    # Store statuses
                    for status in statuses:
                        self.store.upsert_player_status(status)
                        stats['statuses_updated'] += 1
                    
                    # Log successful fetch
                    payload_hash = hashlib.sha256(
                        f"{game.game_id}:{datetime.utcnow().isoformat()}".encode()
                    ).hexdigest()
                    
                    self.store.log_fetch(
                        game_id=game.game_id,
                        source='espn_nba',
                        cache_hit=False,  # Simplified - would check cache
                        payload_hash=payload_hash,
                        success=True
                    )
                
                except Exception as e:
                    error_msg = f"Game {game.game_id}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    stats['errors'].append(error_msg)
                    
                    self.store.log_fetch(
                        game_id=game.game_id,
                        source='espn_nba',
                        cache_hit=False,
                        payload_hash='error',
                        success=False,
                        error_msg=str(e)
                    )
        
        except Exception as e:
            error_msg = f"Scoreboard fetch failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            stats['errors'].append(error_msg)
        
        logger.info(
            f"NBA update complete: {stats['games_found']} games, "
            f"{stats['players_updated']} players, "
            f"{stats['statuses_updated']} statuses, "
            f"{len(stats['errors'])} errors"
        )
        
        return stats
    
    def _fetch_scoreboard(self, target_date: date) -> Dict:
        """Fetch NBA scoreboard for date"""
        date_str = target_date.strftime("%Y%m%d")
        
        return self.client.get(
            f"sports/basketball/nba/scoreboard",
            params={'dates': date_str}
        )
    
    def _parse_games(self, scoreboard: Dict) -> List[Game]:
        """Parse games from scoreboard response"""
        games = []
        
        events = scoreboard.get('events', [])
        
        for event in events:
            try:
                game_id = event['id']
                
                # Parse start time
                date_str = event.get('date')
                start_ts = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                
                # Get teams
                competitions = event.get('competitions', [])
                if not competitions:
                    continue
                
                comp = competitions[0]
                competitors = comp.get('competitors', [])
                
                if len(competitors) != 2:
                    continue
                
                # Identify home/away
                home_team = None
                away_team = None
                
                for competitor in competitors:
                    team_id = competitor['team']['id']
                    home_away = competitor.get('homeAway', '')
                    
                    if home_away == 'home':
                        home_team = team_id
                    else:
                        away_team = team_id
                
                if not home_team or not away_team:
                    # Fallback: first is home, second is away
                    home_team = competitors[0]['team']['id']
                    away_team = competitors[1]['team']['id']
                
                game = Game(
                    game_id=game_id,
                    league=self.LEAGUE,
                    start_ts=start_ts,
                    home_team_id=home_team,
                    away_team_id=away_team
                )
                
                games.append(game)
            
            except Exception as e:
                logger.warning(f"Failed to parse game: {e}")
                continue
        
        return games
    
    def _fetch_team_roster(self, team_id: str) -> List[Player]:
        """Fetch team roster from ESPN"""
        try:
            response = self.client.get(
                f"sports/basketball/nba/teams/{team_id}/roster"
            )
            
            return self._parse_roster(response, team_id)
        
        except ESPNClientError as e:
            logger.error(f"Failed to fetch roster for team {team_id}: {e}")
            return []
    
    def _parse_roster(self, roster_data: Dict, team_id: str) -> List[Player]:
        """Parse roster response"""
        players = []
        
        # ESPN roster structure varies
        # Try multiple paths
        athletes = roster_data.get('athletes', [])
        
        if not athletes:
            # Alternative structure
            entries = roster_data.get('roster', {}).get('entries', [])
            athletes = [e.get('athlete', {}) for e in entries]
        
        for athlete in athletes:
            try:
                player_id = str(athlete.get('id'))
                name = athlete.get('fullName') or athlete.get('displayName', 'Unknown')
                position = athlete.get('position', {})
                
                if isinstance(position, dict):
                    position = position.get('abbreviation', 'N/A')
                
                player = Player(
                    player_id=player_id,
                    league=self.LEAGUE,
                    name=name,
                    team_id=team_id,
                    position=position
                )
                
                players.append(player)
            
            except Exception as e:
                logger.warning(f"Failed to parse player: {e}")
                continue
        
        return players
    
    def _fetch_game_statuses(
        self,
        game: Game,
        players: List[Player]
    ) -> List[PlayerGameStatus]:
        """
        Fetch player statuses for game.
        
        Since ESPN doesn't always have pre-game injuries in scoreboard,
        we infer from recent data and mark as ACTIVE by default.
        """
        statuses = []
        
        for player in players:
            # Check for recent rolling minutes
            minutes_rolling = self.store.get_minutes_rolling(
                self.LEAGUE,
                player.player_id,
                window_n=10
            )
            
            # Default status: ACTIVE
            status_norm = PlayerStatus.ACTIVE
            play_prob = PLAY_PROB_MAP[status_norm]
            est_minutes = minutes_rolling.minutes_avg if minutes_rolling else 20.0
            
            # Infer starter from minutes
            is_starter = None
            if minutes_rolling and minutes_rolling.minutes_avg > 28:
                is_starter = True
            elif minutes_rolling and minutes_rolling.minutes_avg < 15:
                is_starter = False
            
            status = PlayerGameStatus(
                game_id=game.game_id,
                player_id=player.player_id,
                status_norm=status_norm,
                detail="",
                play_prob=play_prob,
                est_minutes=est_minutes,
                is_starter=is_starter,
                ts=datetime.utcnow()
            )
            
            statuses.append(status)
        
        # Try to enhance with injury data if available
        try:
            injuries = self._fetch_injuries()
            self._apply_injury_data(statuses, injuries)
        except Exception as e:
            logger.warning(f"Could not fetch injuries: {e}")
        
        return statuses
    
    def _fetch_injuries(self) -> Dict:
        """
        Fetch current injuries.
        
        Note: ESPN injury endpoint structure varies by league.
        This is a best-effort attempt.
        """
        try:
            return self.client.get("sports/basketball/nba/injuries")
        except:
            return {}
    
    def _apply_injury_data(
        self,
        statuses: List[PlayerGameStatus],
        injury_data: Dict
    ):
        """Apply injury data to statuses (in-place)"""
        # Build injury map
        injury_map = {}
        
        injuries = injury_data.get('injuries', [])
        for injury in injuries:
            try:
                athlete = injury.get('athlete', {})
                player_id = str(athlete.get('id'))
                
                status_text = injury.get('status', 'ACTIVE')
                details = injury.get('details', {})
                detail = details.get('detail', '')
                
                injury_map[player_id] = {
                    'status': status_text,
                    'detail': detail
                }
            except:
                continue
        
        # Apply to statuses
        for status in statuses:
            if status.player_id in injury_map:
                injury_info = injury_map[status.player_id]
                
                # Update status
                status.status_norm = PlayerStatus.from_espn(injury_info['status'])
                status.play_prob = PLAY_PROB_MAP[status.status_norm]
                status.detail = injury_info['detail']
                
                logger.debug(
                    f"Applied injury: {status.player_id} -> {status.status_norm.value}"
                )


# Convenience function
def update_nba_availability(target_date: date) -> Dict:
    """
    Update NBA availability for date.
    
    Usage:
        from pickslab_elite.adapters.availability.espn_nba import update_nba_availability
        stats = update_nba_availability(date(2024, 1, 15))
    """
    adapter = ESPNNBAAdapter()
    return adapter.update_availability_for_date(target_date)
