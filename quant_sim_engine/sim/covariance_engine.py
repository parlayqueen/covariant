"""
Player Interaction Covariance Engine

CRITICAL: Players on the same team are NEVER simulated independently.

This module computes dynamic covariance matrices that capture statistical
dependencies between players:
- Usage competition (negative covariance)
- Assist networks (positive covariance)
- Rebound competition (negative covariance)
- Defensive matchup correlation
- Injury-driven role changes

Matrix recalculates when lineup composition changes.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@dataclass
class PlayerCovarianceProfile:
    """Covariance profile for a player"""
    player_id: str
    usage_rate: float
    assist_rate: float
    rebound_rate: float
    position: str
    minutes_avg: float


class PlayerCovarianceEngine:
    """
    Computes dynamic covariance matrices for player interactions.
    
    RULE: Same-game player outcomes MUST be sampled jointly using this.
    """
    
    # Interaction type weights
    USAGE_COMPETITION_WEIGHT = -0.4  # Negative: usage overlap creates competition
    ASSIST_SYNERGY_WEIGHT = 0.6      # Positive: PG assists → shooter points
    REBOUND_COMPETITION_WEIGHT = -0.3 # Negative: bigs compete for boards
    POSITION_OVERLAP_WEIGHT = -0.2   # Negative: same position competes
    
    def __init__(self, league: str = 'nba'):
        self.league = league
        
        # Position compatibility matrix
        self.position_compatibility = {
            ('PG', 'SG'): 0.3,   # Guards work together
            ('PG', 'SF'): 0.2,
            ('PG', 'PF'): 0.1,
            ('PG', 'C'): 0.0,
            ('SG', 'SF'): 0.2,
            ('SG', 'PF'): 0.1,
            ('SG', 'C'): -0.1,
            ('SF', 'PF'): 0.1,
            ('SF', 'C'): -0.1,
            ('PF', 'C'): -0.2,   # Bigs compete for rebounds
        }
    
    def compute_covariance_matrix(
        self,
        players: List[PlayerCovarianceProfile],
        stat_type: str = 'points'
    ) -> np.ndarray:
        """
        Compute covariance matrix for player group.
        
        Args:
            players: List of player profiles
            stat_type: 'points', 'rebounds', 'assists'
        
        Returns:
            Covariance matrix (n_players × n_players)
        """
        n = len(players)
        
        if n == 0:
            return np.array([[]])
        
        if n == 1:
            # Single player: just variance, no covariance
            return np.array([[self._estimate_variance(players[0], stat_type)]])
        
        # Initialize covariance matrix
        cov_matrix = np.zeros((n, n))
        
        # Compute pairwise covariances
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal: variance
                    cov_matrix[i, j] = self._estimate_variance(players[i], stat_type)
                else:
                    # Off-diagonal: covariance
                    cov_matrix[i, j] = self._compute_pairwise_covariance(
                        players[i],
                        players[j],
                        stat_type
                    )
        
        # Ensure positive semi-definite
        cov_matrix = self._ensure_positive_semidefinite(cov_matrix)
        
        logger.info(
            f"Computed covariance matrix for {n} players ({stat_type}): "
            f"mean off-diagonal={np.mean(cov_matrix[~np.eye(n, dtype=bool)]):.3f}"
        )
        
        return cov_matrix
    
    def _compute_pairwise_covariance(
        self,
        player_i: PlayerCovarianceProfile,
        player_j: PlayerCovarianceProfile,
        stat_type: str
    ) -> float:
        """
        Compute covariance between two players.
        
        Sources of covariance:
        1. Usage competition (negative)
        2. Assist synergy (positive for PG→shooter)
        3. Rebound competition (negative for bigs)
        4. Position overlap (negative)
        """
        # Base variance for scaling
        var_i = self._estimate_variance(player_i, stat_type)
        var_j = self._estimate_variance(player_j, stat_type)
        base_std = np.sqrt(var_i * var_j)
        
        # Start with zero correlation
        correlation = 0.0
        
        # 1. Usage competition
        if stat_type == 'points':
            usage_overlap = min(player_i.usage_rate, player_j.usage_rate) / 100.0
            correlation += self.USAGE_COMPETITION_WEIGHT * usage_overlap
        
        # 2. Assist synergy
        if stat_type == 'points':
            # If i is PG and j is shooter, positive correlation
            if 'PG' in player_i.position and player_j.usage_rate > 15:
                assist_factor = player_i.assist_rate / 100.0
                correlation += self.ASSIST_SYNERGY_WEIGHT * assist_factor * 0.5
        
        if stat_type == 'assists':
            # PG assists positively correlated with team scoring
            if 'PG' in player_i.position and 'PG' in player_j.position:
                correlation += -0.2  # Two PGs compete for assists
        
        # 3. Rebound competition
        if stat_type == 'rebounds':
            if ('C' in player_i.position or 'PF' in player_i.position) and \
               ('C' in player_j.position or 'PF' in player_j.position):
                rebound_overlap = min(player_i.rebound_rate, player_j.rebound_rate) / 100.0
                correlation += self.REBOUND_COMPETITION_WEIGHT * rebound_overlap
        
        # 4. Position compatibility
        pos_key = tuple(sorted([player_i.position[:2], player_j.position[:2]]))
        pos_compat = self.position_compatibility.get(pos_key, 0.0)
        correlation += pos_compat * 0.3
        
        # 5. Minutes overlap adjustment
        minutes_overlap = min(player_i.minutes_avg, player_j.minutes_avg) / 48.0
        correlation *= minutes_overlap
        
        # Convert correlation to covariance
        covariance = correlation * base_std
        
        # Cap covariance magnitude
        max_cov = 0.5 * base_std
        covariance = np.clip(covariance, -max_cov, max_cov)
        
        return covariance
    
    def _estimate_variance(
        self,
        player: PlayerCovarianceProfile,
        stat_type: str
    ) -> float:
        """Estimate variance for player's stat"""
        # Base variance by stat type and usage
        if stat_type == 'points':
            # High usage players have higher variance
            base_var = (player.usage_rate / 20.0) ** 2 * 25
        elif stat_type == 'rebounds':
            base_var = (player.rebound_rate / 15.0) ** 2 * 16
        elif stat_type == 'assists':
            base_var = (player.assist_rate / 20.0) ** 2 * 9
        else:
            base_var = 9.0
        
        # Minutes adjustment
        minutes_factor = player.minutes_avg / 32.0
        base_var *= minutes_factor
        
        return max(1.0, base_var)
    
    def _ensure_positive_semidefinite(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Ensure covariance matrix is positive semi-definite.
        
        Uses eigenvalue decomposition and clipping.
        """
        # Symmetrize
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Clip negative eigenvalues to small positive
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        
        # Reconstruct
        cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return cov_matrix
    
    def compute_correlation_matrix(
        self,
        players: List[PlayerCovarianceProfile],
        stat_type: str = 'points'
    ) -> np.ndarray:
        """
        Compute correlation matrix (normalized covariance).
        
        Returns:
            Correlation matrix with 1.0 on diagonal
        """
        cov = self.compute_covariance_matrix(players, stat_type)
        
        # Convert to correlation
        std_devs = np.sqrt(np.diag(cov))
        
        # Avoid division by zero
        std_devs = np.where(std_devs > 0, std_devs, 1)
        
        # Compute correlation
        corr = cov / np.outer(std_devs, std_devs)
        
        return corr
    
    def update_covariance_for_injury(
        self,
        cov_matrix: np.ndarray,
        injured_player_idx: int,
        remaining_players: List[PlayerCovarianceProfile]
    ) -> np.ndarray:
        """
        Update covariance matrix when a player is injured.
        
        Remaining players have:
        - Increased variance (more responsibility)
        - Weakened correlations (less predictable)
        """
        n = len(remaining_players)
        
        # Remove injured player
        mask = np.ones(cov_matrix.shape[0], dtype=bool)
        mask[injured_player_idx] = False
        
        updated_cov = cov_matrix[mask][:, mask]
        
        # Inflate variance for remaining players (uncertainty from role changes)
        for i in range(n):
            updated_cov[i, i] *= 1.25  # 25% variance increase
        
        # Weaken correlations (more noise)
        for i in range(n):
            for j in range(n):
                if i != j:
                    updated_cov[i, j] *= 0.85  # Weaken by 15%
        
        return updated_cov


def create_covariance_matrix(
    players: List[PlayerCovarianceProfile],
    stat_type: str = 'points',
    league: str = 'nba'
) -> np.ndarray:
    """
    Convenience function to create covariance matrix.
    
    Usage:
        cov = create_covariance_matrix(players, 'points')
    """
    engine = PlayerCovarianceEngine(league)
    return engine.compute_covariance_matrix(players, stat_type)

# Backwards-compatible alias


__all__ = ['PlayerCovarianceEngine', 'CovarianceEngine']

# Backwards-compatible alias

# Backwards-compatible alias

# Backwards-compatible alias
CovarianceEngine = PlayerCovarianceEngine
