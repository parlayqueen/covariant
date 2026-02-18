"""
Usage Redistribution Engine

When a player is injured/out/limited, their usage doesn't disappear - it flows to teammates.

This module models usage redistribution based on:
- Role similarity
- Historical replacement patterns
- Position overlap
- Coaching tendencies

Redistribution affects BOTH mean (increased production) AND variance (role uncertainty).
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class PlayerRole(Enum):
    """Player role classification"""
    PRIMARY_SCORER = "primary_scorer"
    SECONDARY_SCORER = "secondary_scorer"
    FLOOR_GENERAL = "floor_general"
    THREE_POINT_SPECIALIST = "three_point_specialist"
    RIM_PROTECTOR = "rim_protector"
    REBOUNDER = "rebounder"
    DEFENDER = "defender"
    BENCH_SCORER = "bench_scorer"
    ROLE_PLAYER = "role_player"


@dataclass
class UsageProfile:
    """Player's usage profile"""
    player_id: str
    name: str
    base_usage_rate: float  # Normal usage %
    position: str
    role: PlayerRole
    minutes_avg: float
    
    # Skill ratings (0-1)
    scoring_skill: float
    playmaking_skill: float
    rebounding_skill: float


@dataclass
class UsageRedistribution:
    """Usage redistribution outcome"""
    player_id: str
    original_usage: float
    added_usage: float
    new_usage: float
    
    # Statistical impacts
    mean_increase: float  # Expected stat increase
    variance_increase: float  # Uncertainty increase
    confidence: float  # 0-1, how confident in this redistribution


class UsageRedistributionEngine:
    """
    Models usage flow when players are unavailable.
    
    Key principle: Missing usage → redistributed by similarity + skill + coaching
    
    Effects:
    - Increase remaining players' mean production
    - Increase variance (role change uncertainty)
    """
    
    # Role similarity matrix (how similar are roles)
    ROLE_SIMILARITY = {
        (PlayerRole.PRIMARY_SCORER, PlayerRole.SECONDARY_SCORER): 0.8,
        (PlayerRole.PRIMARY_SCORER, PlayerRole.BENCH_SCORER): 0.6,
        (PlayerRole.SECONDARY_SCORER, PlayerRole.BENCH_SCORER): 0.7,
        (PlayerRole.FLOOR_GENERAL, PlayerRole.PRIMARY_SCORER): 0.5,
        (PlayerRole.THREE_POINT_SPECIALIST, PlayerRole.SECONDARY_SCORER): 0.6,
    }
    
    # Position overlap weights
    POSITION_OVERLAP = {
        ('PG', 'PG'): 1.0,
        ('PG', 'SG'): 0.7,
        ('SG', 'SF'): 0.7,
        ('SF', 'PF'): 0.6,
        ('PF', 'C'): 0.7,
        ('C', 'C'): 1.0,
    }
    
    def __init__(self, league: str = 'nba'):
        self.league = league
    
    def redistribute_usage(
        self,
        missing_player: UsageProfile,
        available_players: List[UsageProfile],
        missing_minutes: float = None
    ) -> List[UsageRedistribution]:
        """
        Redistribute usage from missing player to available teammates.
        
        Args:
            missing_player: Player who is out
            available_players: Teammates who will absorb usage
            missing_minutes: Minutes missing player would have played
        
        Returns:
            List of usage redistributions
        """
        if not available_players:
            logger.warning("No available players to redistribute usage")
            return []
        
        # Missing usage to redistribute
        missing_usage = missing_player.base_usage_rate
        
        if missing_minutes is None:
            missing_minutes = missing_player.minutes_avg
        
        logger.info(
            f"Redistributing {missing_usage:.1f}% usage from {missing_player.name} "
            f"({missing_minutes:.1f} min)"
        )
        
        # Compute redistribution weights for each available player
        weights = self._compute_redistribution_weights(
            missing_player,
            available_players
        )
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            # Fallback: equal distribution
            weights = {p.player_id: 1.0 for p in available_players}
            total_weight = len(available_players)
        
        normalized_weights = {
            pid: w / total_weight for pid, w in weights.items()
        }
        
        # Allocate usage
        redistributions = []
        
        for player in available_players:
            weight = normalized_weights.get(player.player_id, 0.0)
            added_usage = missing_usage * weight
            
            # Minutes factor (more minutes = more usage increase)
            minutes_factor = player.minutes_avg / 36.0  # Normalize to 36 min
            
            # Statistical impact
            mean_increase = self._estimate_stat_increase(
                player,
                added_usage,
                minutes_factor
            )
            
            # Variance increase (uncertainty from role change)
            variance_increase = self._estimate_variance_increase(
                player,
                added_usage,
                weight
            )
            
            # Confidence in redistribution
            confidence = self._compute_confidence(
                player,
                missing_player,
                weight
            )
            
            redist = UsageRedistribution(
                player_id=player.player_id,
                original_usage=player.base_usage_rate,
                added_usage=added_usage,
                new_usage=player.base_usage_rate + added_usage,
                mean_increase=mean_increase,
                variance_increase=variance_increase,
                confidence=confidence
            )
            
            redistributions.append(redist)
            
            logger.debug(
                f"  → {player.name}: +{added_usage:.1f}% usage "
                f"(mean +{mean_increase:.1f}, var +{variance_increase:.1f})"
            )
        
        # Verify total adds up
        total_redistributed = sum(r.added_usage for r in redistributions)
        
        logger.info(
            f"Redistributed {total_redistributed:.1f}% / {missing_usage:.1f}% "
            f"({total_redistributed/missing_usage*100:.0f}%)"
        )
        
        return redistributions
    
    def _compute_redistribution_weights(
        self,
        missing: UsageProfile,
        available: List[UsageProfile]
    ) -> Dict[str, float]:
        """
        Compute how much each player should receive.
        
        Based on:
        - Role similarity
        - Position overlap
        - Skill match
        - Current usage (players with more usage can absorb more)
        """
        weights = {}
        
        for player in available:
            weight = 0.0
            
            # 1. Role similarity
            role_sim = self._get_role_similarity(missing.role, player.role)
            weight += role_sim * 0.35
            
            # 2. Position overlap
            pos_overlap = self._get_position_overlap(missing.position, player.position)
            weight += pos_overlap * 0.25
            
            # 3. Skill match
            skill_match = self._compute_skill_match(missing, player)
            weight += skill_match * 0.25
            
            # 4. Current usage capacity
            # Players with higher usage can absorb more
            usage_capacity = min(1.0, player.base_usage_rate / 20.0)
            weight += usage_capacity * 0.15
            
            # Cap weight
            weight = min(1.0, weight)
            
            weights[player.player_id] = weight
        
        return weights
    
    def _get_role_similarity(
        self,
        role1: PlayerRole,
        role2: PlayerRole
    ) -> float:
        """Get similarity between two roles"""
        if role1 == role2:
            return 1.0
        
        # Check both orderings
        sim = self.ROLE_SIMILARITY.get((role1, role2))
        if sim is not None:
            return sim
        
        sim = self.ROLE_SIMILARITY.get((role2, role1))
        if sim is not None:
            return sim
        
        # Default: low similarity
        return 0.2
    
    def _get_position_overlap(self, pos1: str, pos2: str) -> float:
        """Get overlap between positions"""
        # Extract primary position (first 2 chars)
        p1 = pos1[:2] if len(pos1) >= 2 else pos1
        p2 = pos2[:2] if len(pos2) >= 2 else pos2
        
        # Check both orderings
        overlap = self.POSITION_OVERLAP.get((p1, p2))
        if overlap is not None:
            return overlap
        
        overlap = self.POSITION_OVERLAP.get((p2, p1))
        if overlap is not None:
            return overlap
        
        # Same position always high overlap
        if p1 == p2:
            return 1.0
        
        # Default: moderate overlap
        return 0.4
    
    def _compute_skill_match(
        self,
        missing: UsageProfile,
        available: UsageProfile
    ) -> float:
        """Compute skill match between players"""
        # Compare relevant skills
        scoring_match = 1 - abs(missing.scoring_skill - available.scoring_skill)
        playmaking_match = 1 - abs(missing.playmaking_skill - available.playmaking_skill)
        
        # Weight by role
        if missing.role in [PlayerRole.PRIMARY_SCORER, PlayerRole.SECONDARY_SCORER]:
            match = scoring_match * 0.7 + playmaking_match * 0.3
        elif missing.role == PlayerRole.FLOOR_GENERAL:
            match = playmaking_match * 0.7 + scoring_match * 0.3
        else:
            match = (scoring_match + playmaking_match) / 2
        
        return match
    
    def _estimate_stat_increase(
        self,
        player: UsageProfile,
        added_usage: float,
        minutes_factor: float
    ) -> float:
        """
        Estimate statistical increase for player.
        
        More usage → more production, but with diminishing returns.
        """
        # Base: linear relationship
        # ~1% usage = ~0.5 points per game
        base_increase = added_usage * 0.5
        
        # Adjust for skill (better players convert usage more efficiently)
        skill_factor = (player.scoring_skill + player.playmaking_skill) / 2
        base_increase *= (0.7 + skill_factor * 0.6)
        
        # Minutes factor
        base_increase *= minutes_factor
        
        # Diminishing returns (hard to absorb >10% usage increase)
        if added_usage > 10:
            excess = added_usage - 10
            base_increase -= excess * 0.2
        
        return max(0, base_increase)
    
    def _estimate_variance_increase(
        self,
        player: UsageProfile,
        added_usage: float,
        weight: float
    ) -> float:
        """
        Estimate variance increase from role change.
        
        More usage → more uncertainty (new role).
        """
        # Base variance increase proportional to usage change
        base_var_increase = (added_usage / 10.0) ** 2 * 4.0
        
        # Increase if player not used to high usage
        if player.base_usage_rate < 20:
            inexperience_factor = 1.0 + (20 - player.base_usage_rate) / 20.0
            base_var_increase *= inexperience_factor
        
        # Weight factor (receiving more = more uncertain)
        base_var_increase *= (1 + weight * 0.5)
        
        return base_var_increase
    
    def _compute_confidence(
        self,
        player: UsageProfile,
        missing: UsageProfile,
        weight: float
    ) -> float:
        """
        Compute confidence in this redistribution.
        
        Higher when:
        - Similar roles
        - Similar skills
        - Player experienced with high usage
        """
        confidence = 0.5  # Base
        
        # Role similarity boost
        role_sim = self._get_role_similarity(player.role, missing.role)
        confidence += role_sim * 0.2
        
        # Skill match boost
        skill_match = self._compute_skill_match(missing, player)
        confidence += skill_match * 0.2
        
        # Experience with high usage
        if player.base_usage_rate > 20:
            confidence += 0.1
        
        return min(1.0, confidence)


def redistribute_missing_usage(
    injured_player: UsageProfile,
    available_teammates: List[UsageProfile]
) -> List[UsageRedistribution]:
    """
    Convenience function for usage redistribution.
    
    Usage:
        redistributions = redistribute_missing_usage(
            injured_lebron,
            [ad, dlo, reaves, ...]
        )
        
        for r in redistributions:
            print(f"{r.player_id}: +{r.added_usage:.1f}% usage")
    """
    engine = UsageRedistributionEngine()
    return engine.redistribute_usage(injured_player, available_teammates)
