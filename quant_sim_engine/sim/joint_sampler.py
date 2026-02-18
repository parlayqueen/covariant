"""
Multivariate Monte Carlo Joint Sampler

CRITICAL RULE: Players sharing court time MUST be sampled jointly.

This module replaces independent sampling with correlated multivariate sampling:
- Uses covariance matrices from covariance_engine
- Samples from multivariate normal or copula distributions
- Ensures realistic joint outcomes
- Enables correlated prop analysis

Independent player sampling is PROHIBITED.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import multivariate_normal, norm
from scipy.linalg import cholesky

logger = logging.getLogger(__name__)


@dataclass
class PlayerDistribution:
    """Statistical distribution for a player"""
    player_id: str
    mean: float
    std: float
    min_value: float = 0.0  # Floor (can't be negative)
    max_value: Optional[float] = None  # Optional ceiling


class MultivariateJointSampler:
    """
    Samples correlated player outcomes using multivariate distributions.
    
    Methods:
    - Multivariate Normal (fast, works well for most stats)
    - Copula-based (more flexible for non-normal marginals)
    """
    
    def __init__(self, method: str = 'normal'):
        """
        Initialize sampler.
        
        Args:
            method: 'normal' or 'copula'
        """
        self.method = method
        
        if method not in ['normal', 'copula']:
            raise ValueError(f"Method must be 'normal' or 'copula', got {method}")
    
    def sample_joint(
        self,
        distributions: List[PlayerDistribution],
        covariance_matrix: np.ndarray,
        n_simulations: int = 10000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample joint player outcomes.
        
        Args:
            distributions: List of player distributions
            covariance_matrix: Covariance matrix (n_players × n_players)
            n_simulations: Number of Monte Carlo simulations
            seed: Random seed for reproducibility
        
        Returns:
            Samples array (n_simulations × n_players)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_players = len(distributions)
        
        if n_players == 0:
            return np.array([])
        
        # Validate covariance matrix shape
        if covariance_matrix.shape != (n_players, n_players):
            raise ValueError(
                f"Covariance matrix shape {covariance_matrix.shape} "
                f"doesn't match {n_players} players"
            )
        
        # Extract means and stds
        means = np.array([d.mean for d in distributions])
        
        # Sample based on method
        if self.method == 'normal':
            samples = self._sample_multivariate_normal(
                means,
                covariance_matrix,
                n_simulations
            )
        else:  # copula
            samples = self._sample_copula(
                distributions,
                covariance_matrix,
                n_simulations
            )
        
        # Apply bounds
        for i, dist in enumerate(distributions):
            # Floor at minimum
            samples[:, i] = np.maximum(samples[:, i], dist.min_value)
            
            # Cap at maximum if specified
            if dist.max_value is not None:
                samples[:, i] = np.minimum(samples[:, i], dist.max_value)
        
        logger.info(
            f"Sampled {n_simulations} joint outcomes for {n_players} players "
            f"using {self.method} method"
        )
        
        return samples
    
    def _sample_multivariate_normal(
        self,
        means: np.ndarray,
        covariance: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Sample from multivariate normal distribution.
        
        Fast and works well for most sports stats.
        """
        # Use Cholesky decomposition for efficiency
        try:
            L = cholesky(covariance, lower=True)
        except np.linalg.LinAlgError:
            # Covariance not positive definite - add small ridge
            logger.warning("Covariance matrix not PSD, adding ridge")
            ridge = np.eye(covariance.shape[0]) * 1e-6
            L = cholesky(covariance + ridge, lower=True)
        
        # Generate standard normal samples
        z = np.random.randn(n_samples, len(means))
        
        # Transform to correlated samples
        samples = means + z @ L.T
        
        return samples
    
    def _sample_copula(
        self,
        distributions: List[PlayerDistribution],
        covariance: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Sample using Gaussian copula.
        
        More flexible - allows non-normal marginal distributions.
        """
        n_players = len(distributions)
        
        # Convert covariance to correlation
        stds = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(stds, stds)
        
        # Ensure valid correlation matrix
        correlation = np.clip(correlation, -0.999, 0.999)
        np.fill_diagonal(correlation, 1.0)
        
        # Sample from Gaussian copula (uniform marginals)
        try:
            L = cholesky(correlation, lower=True)
        except np.linalg.LinAlgError:
            logger.warning("Correlation matrix not PSD, using fallback")
            L = np.eye(n_players)
        
        z = np.random.randn(n_samples, n_players)
        u = norm.cdf(z @ L.T)  # Uniform marginals
        
        # Transform to player-specific distributions
        samples = np.zeros((n_samples, n_players))
        
        for i, dist in enumerate(distributions):
            # Transform uniform to normal with player's mean/std
            samples[:, i] = norm.ppf(u[:, i]) * dist.std + dist.mean
        
        return samples
    
    def compute_joint_probability(
        self,
        outcomes: Dict[str, float],
        distributions: List[PlayerDistribution],
        covariance_matrix: np.ndarray
    ) -> float:
        """
        Compute joint probability of specific outcomes.
        
        Used for parlay EV calculations.
        
        Args:
            outcomes: Dict of player_id -> outcome value
            distributions: Player distributions
            covariance_matrix: Covariance matrix
        
        Returns:
            Joint probability
        """
        # Create outcome vector
        outcome_vec = np.array([
            outcomes.get(d.player_id, d.mean)
            for d in distributions
        ])
        
        # Means vector
        means = np.array([d.mean for d in distributions])
        
        # Compute multivariate normal PDF
        try:
            pdf = multivariate_normal.pdf(
                outcome_vec,
                mean=means,
                cov=covariance_matrix
            )
        except:
            # Fallback: product of independent probabilities
            logger.warning("Joint probability failed, using independent fallback")
            pdf = np.prod([
                norm.pdf(outcome_vec[i], loc=means[i], scale=np.sqrt(covariance_matrix[i, i]))
                for i in range(len(distributions))
            ])
        
        return pdf
    
    def estimate_parlay_probability(
        self,
        prop_outcomes: List[Tuple[str, str, float]],
        distributions: List[PlayerDistribution],
        covariance_matrix: np.ndarray,
        n_simulations: int = 100000
    ) -> float:
        """
        Estimate parlay probability via simulation.
        
        Args:
            prop_outcomes: List of (player_id, comparison, value)
                         e.g., [('player_1', 'over', 25.5), ...]
            distributions: Player distributions
            covariance_matrix: Covariance matrix
            n_simulations: Number of simulations
        
        Returns:
            Estimated probability of parlay hitting
        """
        # Sample joint outcomes
        samples = self.sample_joint(
            distributions,
            covariance_matrix,
            n_simulations
        )
        
        # Map player IDs to indices
        player_to_idx = {d.player_id: i for i, d in enumerate(distributions)}
        
        # Check each simulation
        hits = 0
        
        for sim in samples:
            all_hit = True
            
            for player_id, comparison, value in prop_outcomes:
                idx = player_to_idx.get(player_id)
                if idx is None:
                    continue
                
                player_outcome = sim[idx]
                
                if comparison == 'over':
                    if player_outcome <= value:
                        all_hit = False
                        break
                elif comparison == 'under':
                    if player_outcome >= value:
                        all_hit = False
                        break
            
            if all_hit:
                hits += 1
        
        probability = hits / n_simulations
        
        logger.info(
            f"Parlay probability: {probability:.4f} "
            f"({hits}/{n_simulations} simulations)"
        )
        
        return probability


def sample_correlated_players(
    players: List[PlayerDistribution],
    covariance: np.ndarray,
    n_sims: int = 10000,
    method: str = 'normal'
) -> np.ndarray:
    """
    Convenience function for joint sampling.

    If caller passes a *correlation* matrix (diag ~ 1), we scale it into
    a true covariance matrix using each player's std dev:
        Sigma = D @ Corr @ D
    """
    # --- corr_to_cov ---
    try:
        diag = np.diag(covariance).astype(float)
        if np.all(np.isfinite(diag)) and np.all(np.abs(diag - 1.0) < 1e-6):
            sig = np.array([float(getattr(d, 'std', 1.0)) for d in players], dtype=float)
            D = np.diag(sig)
            covariance = D @ covariance @ D
    except Exception:
        pass
    # --- end corr_to_cov ---

    sampler = MultivariateJointSampler(method=method)
    return sampler.sample_joint(players, covariance, n_sims)


def compute_parlay_ev(
    prop_bets: List[Tuple[str, str, float, float]],
    distributions: List[PlayerDistribution],
    covariance: np.ndarray,
    n_sims: int = 100000
) -> Dict[str, float]:
    """
    Compute parlay expected value accounting for correlation.
    
    Args:
        prop_bets: List of (player_id, comparison, line, odds)
        distributions: Player distributions
        covariance: Covariance matrix
        n_sims: Simulations
    
    Returns:
        Dict with 'probability', 'ev', 'independent_ev' for comparison
    """
    sampler = MultivariateJointSampler()
    
    # Props without odds
    props = [(p, c, l) for p, c, l, _ in prop_bets]
    
    # Correlated probability
    corr_prob = sampler.estimate_parlay_probability(
        props,
        distributions,
        covariance,
        n_sims
    )
    
    # Independent probability (for comparison)
    indep_prob = 1.0
    for player_id, comparison, line, _ in prop_bets:
        # Find player distribution
        dist = next((d for d in distributions if d.player_id == player_id), None)
        if dist:
            if comparison == 'over':
                p = 1 - norm.cdf(line, loc=dist.mean, scale=dist.std)
            else:  # under
                p = norm.cdf(line, loc=dist.mean, scale=dist.std)
            indep_prob *= p
    
    # Compute EV (assuming American odds)
    # Combine all odds for parlay payout
    total_odds_multiplier = 1.0
    for _, _, _, odds in prop_bets:
        if odds > 0:
            total_odds_multiplier *= (1 + odds / 100)
        else:
            total_odds_multiplier *= (1 + 100 / abs(odds))
    
    # EV = probability * payout - 1
    corr_ev = corr_prob * total_odds_multiplier - 1
    indep_ev = indep_prob * total_odds_multiplier - 1
    
    return {
        'correlated_probability': corr_prob,
        'independent_probability': indep_prob,
        'correlation_adjustment': corr_prob - indep_prob,
        'correlated_ev': corr_ev,
        'independent_ev': indep_ev,
        'ev_difference': corr_ev - indep_ev
    }
