"""
Player Impact Engine

Derives player tiers and impact from REAL ESPN data:
- Minutes from rolling averages (NO FAKE METRICS)
- Tiers from team-relative minutes percentiles
- Impact computed with uncertainty

NO MOCK DATA.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from pickslab_elite.core.store.availability_store import (
    get_availability_store,
    PlayerGameStatus, PlayerMinutesRolling,
    PlayerStatus, Player
)


logger = logging.getLogger(__name__)


class PlayerTier(Enum):
    """Player tier classification"""
    T0_SUPERSTAR = 0  # Top 1 on team
    T1_STAR = 1       # Next 2
    T2_STARTER = 2    # Next 5
    T3_ROTATION = 3   # Next 5
    T4_BENCH = 4      # Rest


# League-specific impact values (points to spread equivalent)
TIER_IMPACT_BASE = {
    'nba': {
        PlayerTier.T0_SUPERSTAR: 8.0,
        PlayerTier.T1_STAR: 4.5,
        PlayerTier.T2_STARTER: 2.5,
        PlayerTier.T3_ROTATION: 1.0,
        PlayerTier.T4_BENCH: 0.3,
    },
    'nfl': {
        PlayerTier.T0_SUPERSTAR: 6.5,
        PlayerTier.T1_STAR: 3.5,
        PlayerTier.T2_STARTER: 2.0,
        PlayerTier.T3_ROTATION: 0.8,
        PlayerTier.T4_BENCH: 0.2,
    },
}

# Starter minutes by league (used for minute_factor calculation)
STARTER_MINUTES = {
    'nba': 32.0,
    'nfl': 60.0,  # snaps
}


@dataclass
class PlayerImpactAssessment:
    """Complete impact assessment for a player"""
    player_id: str
    player_name: str
    tier: PlayerTier
    est_minutes: float
    is_starter: Optional[bool]
    status: PlayerStatus
    play_prob: float
    
    # Computed impact
    base_impact: float
    minute_factor: float
    adjusted_impact: float
    mean_contribution: float  # Expected contribution to margin
    variance_contribution: float  # Added uncertainty


class PlayerImpactEngine:
    """
    Computes player impact from REAL data only.
    
    NO FAKE PLAYER STATS. Everything derived from ESPN minutes history.
    """
    
    def __init__(self, league: str = 'nba'):
        self.league = league
        self.store = get_availability_store()
        
        # Get league-specific config
        self.tier_impacts = TIER_IMPACT_BASE.get(league, TIER_IMPACT_BASE['nba'])
        self.starter_minutes = STARTER_MINUTES.get(league, 32.0)
    
    def assess_team_impact(
        self,
        team_id: str,
        game_id: str
    ) -> Tuple[List[PlayerImpactAssessment], float, float]:
        """
        Assess total team impact from availability.
        
        Args:
            team_id: Team ID
            game_id: Game ID
        
        Returns:
            (assessments, total_mean_adj, total_variance_adj)
        """
        # Get team players
        players = self.store.get_team_players(self.league, team_id)
        
        if not players:
            logger.warning(f"No players found for team {team_id}")
            return [], 0.0, 0.0
        
        # Compute tiers from minutes
        player_tiers = self._compute_tiers(players)
        
        # Get statuses
        statuses = {
            s.player_id: s
            for s in self.store.get_game_statuses(game_id)
        }
        
        # Assess each player
        assessments = []
        
        for player in players:
            status = statuses.get(player.player_id)
            
            if not status:
                # No status - assume active
                logger.debug(f"No status for {player.name}, assuming active")
                continue
            
            assessment = self._assess_player(player, player_tiers, status)
            assessments.append(assessment)
        
        # Aggregate impact
        total_mean = sum(a.mean_contribution for a in assessments)
        total_variance = sum(a.variance_contribution for a in assessments)
        
        logger.info(
            f"Team {team_id}: {len(assessments)} players assessed, "
            f"mean_adj={total_mean:.2f}, var_adj={total_variance:.2f}"
        )
        
        return assessments, total_mean, total_variance
    
    def _compute_tiers(self, players: List[Player]) -> Dict[str, PlayerTier]:
        """
        Compute tiers from rolling minutes (REAL DATA).
        
        Ranking based on team-relative minutes percentiles.
        """
        # Get minutes for all players
        player_minutes = []
        
        for player in players:
            minutes_rolling = self.store.get_minutes_rolling(
                self.league,
                player.player_id,
                window_n=10
            )
            
            if minutes_rolling:
                player_minutes.append((player.player_id, minutes_rolling.minutes_avg))
            else:
                # No history - assume bench
                player_minutes.append((player.player_id, 0.0))
        
        # Sort by minutes descending
        player_minutes.sort(key=lambda x: x[1], reverse=True)
        
        # Assign tiers
        tiers = {}
        
        for i, (player_id, minutes) in enumerate(player_minutes):
            if i == 0:
                tier = PlayerTier.T0_SUPERSTAR
            elif i <= 2:
                tier = PlayerTier.T1_STAR
            elif i <= 7:
                tier = PlayerTier.T2_STARTER
            elif i <= 12:
                tier = PlayerTier.T3_ROTATION
            else:
                tier = PlayerTier.T4_BENCH
            
            tiers[player_id] = tier
        
        return tiers
    
    def _assess_player(
        self,
        player: Player,
        tiers: Dict[str, PlayerTier],
        status: PlayerGameStatus
    ) -> PlayerImpactAssessment:
        """Assess individual player impact"""
        tier = tiers.get(player.player_id, PlayerTier.T4_BENCH)
        
        # Base impact from tier
        base_impact = self.tier_impacts[tier]
        
        # Minute factor
        minute_factor = min(1.25, max(0.15, status.est_minutes / self.starter_minutes))
        
        # Adjusted impact
        adjusted_impact = base_impact * minute_factor
        
        # Absence probability
        missing_prob = 1.0 - status.play_prob
        
        # Expected contribution (negative if likely to miss)
        mean_contribution = -missing_prob * adjusted_impact
        
        # Variance contribution (uncertainty)
        variance_contribution = missing_prob * (1 - missing_prob) * (adjusted_impact ** 2)
        
        # Star-out shock (chemistry disruption)
        if tier in [PlayerTier.T0_SUPERSTAR, PlayerTier.T1_STAR]:
            if status.status_norm in [PlayerStatus.OUT, PlayerStatus.IR, PlayerStatus.SUSPENDED]:
                # Add extra uncertainty
                variance_contribution *= 1.5
                # Add small mean shock
                mean_contribution -= 1.5
        
        return PlayerImpactAssessment(
            player_id=player.player_id,
            player_name=player.name,
            tier=tier,
            est_minutes=status.est_minutes,
            is_starter=status.is_starter,
            status=status.status_norm,
            play_prob=status.play_prob,
            base_impact=base_impact,
            minute_factor=minute_factor,
            adjusted_impact=adjusted_impact,
            mean_contribution=mean_contribution,
            variance_contribution=variance_contribution
        )
