"""
Simulation Adjuster

Applies availability adjustments to matchup parameters.

Single-point integration: call this before running simulations.
"""

import logging
from typing import Dict, Tuple, Optional
from datetime import datetime

from pickslab_elite.core.store.availability_store import get_availability_store, Game
from pickslab_elite.core.impact.player_impact_engine import PlayerImpactEngine, PlayerImpactAssessment


logger = logging.getLogger(__name__)


def apply_availability_adjustments(
    matchup_features: Dict,
    game_id: str,
    league: str,
    confidence_threshold: float = 50.0
) -> Tuple[Dict, Dict]:
    """
    Apply availability adjustments to matchup parameters.
    
    **INTEGRATION POINT**: Call this where you assemble pre-sim parameters.
    
    Args:
        matchup_features: Dict with baseline params (home_strength, away_strength, etc.)
        game_id: Game ID
        league: League ('nba', 'nfl', etc.)
        confidence_threshold: Minimum confidence to apply (0-100)
    
    Returns:
        (adjusted_features, explanation)
        
    Usage:
        # BEFORE simulation
        features = {'home_mu': 105.0, 'away_mu': 103.0, 'home_sd': 12.0, 'away_sd': 12.0}
        
        # APPLY AVAILABILITY
        adj_features, explain = apply_availability_adjustments(features, game_id, 'nba')
        
        # CONTINUE with simulation using adj_features
    """
    store = get_availability_store()
    engine = PlayerImpactEngine(league=league)
    
    # Get game
    game = store.get_game(league, game_id)
    
    if not game:
        logger.warning(f"Game {game_id} not found, no adjustments applied")
        return matchup_features, {'confidence': 0, 'reason': 'game_not_found'}
    
    # Assess both teams
    home_assessments, home_mean, home_var = engine.assess_team_impact(
        game.home_team_id, game_id
    )
    
    away_assessments, away_mean, away_var = engine.assess_team_impact(
        game.away_team_id, game_id
    )
    
    # Compute confidence
    confidence = _compute_confidence(
        game, home_assessments, away_assessments
    )
    
    # Build explanation
    explanation = {
        'game_id': game_id,
        'league': league,
        'confidence': confidence,
        'home_team': game.home_team_id,
        'away_team': game.away_team_id,
        'home_mean_adj': home_mean,
        'away_mean_adj': away_mean,
        'home_var_adj': home_var,
        'away_var_adj': away_var,
        'home_players': [
            {
                'name': a.player_name,
                'tier': a.tier.name,
                'status': a.status.value,
                'play_prob': a.play_prob,
                'mean_contrib': a.mean_contribution,
                'var_contrib': a.variance_contribution
            }
            for a in home_assessments
            if abs(a.mean_contribution) > 0.5 or a.variance_contribution > 1.0
        ],
        'away_players': [
            {
                'name': a.player_name,
                'tier': a.tier.name,
                'status': a.status.value,
                'play_prob': a.play_prob,
                'mean_contrib': a.mean_contribution,
                'var_contrib': a.variance_contribution
            }
            for a in away_assessments
            if abs(a.mean_contribution) > 0.5 or a.variance_contribution > 1.0
        ],
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Check confidence threshold
    if confidence < confidence_threshold:
        logger.warning(
            f"Confidence {confidence:.1f} below threshold {confidence_threshold}, "
            "adjustments applied but flagged"
        )
        explanation['warning'] = 'low_confidence'
    
    # Apply adjustments (with caps)
    adjusted = matchup_features.copy()
    
    # Margin adjustment (home perspective)
    margin_adj = home_mean - away_mean
    margin_adj = _cap_value(margin_adj, -10.0, 10.0)
    
    # Apply to means
    if 'home_mu' in adjusted and 'away_mu' in adjusted:
        adjusted['home_mu'] += margin_adj / 2
        adjusted['away_mu'] -= margin_adj / 2
    
    # Variance adjustment (inflate uncertainty)
    total_var_adj = home_var + away_var
    total_var_adj = _cap_value(total_var_adj, 0, 50.0)
    
    # Apply to standard deviations (add in quadrature)
    if 'home_sd' in adjusted:
        adjusted['home_sd'] = (adjusted['home_sd']**2 + total_var_adj)**0.5
    
    if 'away_sd' in adjusted:
        adjusted['away_sd'] = (adjusted['away_sd']**2 + total_var_adj)**0.5
    
    logger.info(
        f"Adjustments applied: margin={margin_adj:.2f}, var={total_var_adj:.2f}, "
        f"confidence={confidence:.1f}"
    )
    
    return adjusted, explanation


def _compute_confidence(
    game: Game,
    home_assessments,
    away_assessments
) -> float:
    """
    Compute confidence score (0-100).
    
    Penalizes:
    - Stale data
    - Many questionable players
    - Missing roster data
    """
    score = 100.0
    
    # Data freshness (penalize if >6 hours old)
    now = datetime.utcnow()
    age_hours = (now - game.start_ts).total_seconds() / 3600
    
    if age_hours > 6:
        score -= 20
    elif age_hours > 3:
        score -= 10
    
    # Roster completeness
    total_players = len(home_assessments) + len(away_assessments)
    if total_players < 20:
        score -= 30
    
    # Questionable players (uncertainty)
    questionable_count = sum(
        1 for a in home_assessments + away_assessments
        if 0.3 < a.play_prob < 0.7
    )
    
    score -= questionable_count * 5
    
    return max(0, min(100, score))


def _cap_value(value: float, min_val: float, max_val: float) -> float:
    """Cap value to range"""
    return max(min_val, min(max_val, value))
