"""
Elite Simulation Adjuster - Complete Integration

Professional integration layer with:
- Elite player impact profiles
- Multi-source injury verification  
- Advanced caching with staleness detection
- Scenario analysis (best/worst case)
- Performance monitoring
- Confidence scoring

This is the SINGLE INTEGRATION POINT for your existing system.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import time

from pickslab_elite.core.store.availability_store import get_availability_store, Game, PlayerStatus
from pickslab_elite.core.impact.elite_player_impact import (
    ElitePlayerImpactEngine,
    PlayerImpactProfile,
    PlayerTier
)
from pickslab_elite.core.data.enhanced_injury_fetcher import MultiSourceInjuryFetcher


logger = logging.getLogger(__name__)


@dataclass
class DetailedPlayerImpact:
    """Detailed individual player impact for reporting"""
    player_name: str
    tier: str
    status: str
    play_prob: float
    minutes_expected: float
    
    # Impact
    mean_impact: float
    variance_impact: float
    
    # Breakdown
    offensive: float
    defensive: float
    chemistry: float
    
    # Flags
    is_star: bool
    is_starter: bool


@dataclass
class TeamAdjustment:
    """Complete team adjustment"""
    team_id: str
    
    # Player impacts
    players: List[DetailedPlayerImpact]
    
    # Totals
    total_mean_adj: float
    total_variance_adj: float
    
    # By tier
    mvp_impact: float
    superstar_impact: float
    star_impact: float
    
    # Quality
    confidence: float
    
    # Flags
    multiple_stars_out: bool
    starting_lineup_changed: bool


class EliteSimulationAdjuster:
    """
    Elite simulation adjuster with full professional features.
    
    USAGE (Single line integration):
        features, explanation = apply_elite_adjustments(features, game_id, 'nba')
    """
    
    # Optimizations
    CACHE_TTL_SECONDS = 1800  # 30 minutes
    
    # Safety caps (per team)
    MAX_MEAN_ADJUSTMENT = 18.0  # points
    MAX_VARIANCE_ADJUSTMENT = 100.0  # points²
    
    # Confidence thresholds
    CONFIDENCE_MINIMUM = 35.0
    CONFIDENCE_RECOMMEND = 65.0
    
    def __init__(self, league: str = 'nba'):
        self.league = league
        self.store = get_availability_store()
        self.impact_engine = ElitePlayerImpactEngine(league)
        self.injury_fetcher = MultiSourceInjuryFetcher(league)
        
        # Performance tracking
        self.stats = {
            'adjustments_computed': 0,
            'cache_hits': 0,
            'avg_compute_time_ms': 0.0
        }
        
        # Cache
        self._cache: Dict[str, Tuple[TeamAdjustment, float]] = {}
    
    def apply_elite_adjustments(
        self,
        matchup_features: Dict,
        game_id: str,
        include_scenarios: bool = False,
        force_fresh: bool = False
    ) -> Tuple[Dict, Dict]:
        """
        **MAIN INTEGRATION POINT**
        
        Apply elite adjustments to your matchup features.
        
        Args:
            matchup_features: Your baseline features dict
            game_id: ESPN game ID
            include_scenarios: Include best/worst case scenarios
            force_fresh: Skip cache, force fresh computation
        
        Returns:
            (adjusted_features, comprehensive_explanation)
        
        Example:
            features = {'home_mu': 105, 'away_mu': 103, 'home_sd': 12, 'away_sd': 12}
            adj_features, explain = apply_elite_adjustments(features, game_id, 'nba')
            # Continue with adj_features...
        """
        start_time = time.time()
        
        # Get game
        game = self.store.get_game(self.league, game_id)
        
        if not game:
            logger.warning(f"Game {game_id} not found")
            return matchup_features, {'error': 'game_not_found', 'confidence': 0}
        
        # Assess both teams
        home_adj = self._assess_team(
            game.home_team_id,
            game_id,
            use_cache=(not force_fresh)
        )
        
        away_adj = self._assess_team(
            game.away_team_id,
            game_id,
            use_cache=(not force_fresh)
        )
        
        # Apply adjustments
        adjusted_features = self._apply_adjustments(
            matchup_features.copy(),
            home_adj,
            away_adj
        )
        
        # Build explanation
        explanation = self._build_explanation(
            game,
            home_adj,
            away_adj,
            matchup_features,
            adjusted_features
        )
        
        # Scenarios if requested
        if include_scenarios:
            explanation['scenarios'] = self._compute_scenarios(
                home_adj,
                away_adj
            )
        
        # Performance tracking
        compute_time_ms = (time.time() - start_time) * 1000
        self.stats['adjustments_computed'] += 1
        self.stats['avg_compute_time_ms'] = (
            (self.stats['avg_compute_time_ms'] * (self.stats['adjustments_computed'] - 1) +
             compute_time_ms) / self.stats['adjustments_computed']
        )
        
        explanation['performance'] = {
            'compute_time_ms': compute_time_ms,
            'cache_used': not force_fresh
        }
        
        logger.info(
            f"Adjustments applied for {game_id}: "
            f"home={home_adj.total_mean_adj:.2f}, away={away_adj.total_mean_adj:.2f}, "
            f"confidence={explanation['confidence']:.0f}, time={compute_time_ms:.1f}ms"
        )
        
        return adjusted_features, explanation
    
    def _assess_team(
        self,
        team_id: str,
        game_id: str,
        use_cache: bool
    ) -> TeamAdjustment:
        """Assess team with elite caching"""
        cache_key = f"{game_id}:{team_id}"
        
        # Check cache
        if use_cache and cache_key in self._cache:
            cached_adj, cache_time = self._cache[cache_key]
            
            age = time.time() - cache_time
            if age < self.CACHE_TTL_SECONDS:
                self.stats['cache_hits'] += 1
                logger.debug(f"Cache HIT for {team_id} (age: {age:.0f}s)")
                return cached_adj
        
        # Compute fresh
        adjustment = self._compute_team_adjustment(team_id, game_id)
        
        # Cache it
        self._cache[cache_key] = (adjustment, time.time())
        
        return adjustment
    
    def _compute_team_adjustment(
        self,
        team_id: str,
        game_id: str
    ) -> TeamAdjustment:
        """Compute complete team adjustment using elite engine"""
        # Get players
        players = self.store.get_team_players(self.league, team_id)
        
        if not players:
            return self._empty_adjustment(team_id)
        
        # Build impact profiles for all players
        player_impacts = []
        
        for player in players:
            # Get status
            status = self.store.get_player_status(game_id, player.player_id)
            
            if not status:
                continue
            
            # Build profile
            profile = self.impact_engine.build_player_profile(player)
            
            # Compute game impact
            mean_impact, var_impact, breakdown = self.impact_engine.compute_game_impact(
                profile,
                status.status_norm,
                status.play_prob,
                status.est_minutes
            )
            
            # Create detailed record
            detailed = DetailedPlayerImpact(
                player_name=player.name,
                tier=profile.tier.name,
                status=status.status_norm.value,
                play_prob=status.play_prob,
                minutes_expected=status.est_minutes,
                mean_impact=mean_impact,
                variance_impact=var_impact,
                offensive=breakdown['offensive_component'] if 'offensive_component' in breakdown else 0,
                defensive=breakdown['defensive_component'] if 'defensive_component' in breakdown else 0,
                chemistry=breakdown.get('chemistry_penalty', 0),
                is_star=(profile.tier.value <= 2),
                is_starter=(status.is_starter or False)
            )
            
            player_impacts.append(detailed)
        
        # Aggregate
        total_mean = sum(p.mean_impact for p in player_impacts)
        total_variance = sum(p.variance_impact for p in player_impacts)
        
        # Apply safety caps
        total_mean = np.clip(total_mean, -self.MAX_MEAN_ADJUSTMENT, self.MAX_MEAN_ADJUSTMENT)
        total_variance = np.clip(total_variance, 0, self.MAX_VARIANCE_ADJUSTMENT)
        
        # Tier breakdowns
        mvp_impact = sum(
            p.mean_impact for p in player_impacts
            if 'T0_MVP' in p.tier
        )
        
        superstar_impact = sum(
            p.mean_impact for p in player_impacts
            if 'T1_SUPERSTAR' in p.tier
        )
        
        star_impact = sum(
            p.mean_impact for p in player_impacts
            if 'T2_STAR' in p.tier
        )
        
        # Compute confidence
        confidence = self._compute_confidence(player_impacts)
        
        # Flags
        stars_out = sum(1 for p in player_impacts if p.is_star and p.play_prob < 0.5)
        lineup_changed = any(p.is_starter and p.play_prob < 0.9 for p in player_impacts)
        
        return TeamAdjustment(
            team_id=team_id,
            players=player_impacts,
            total_mean_adj=total_mean,
            total_variance_adj=total_variance,
            mvp_impact=mvp_impact,
            superstar_impact=superstar_impact,
            star_impact=star_impact,
            confidence=confidence,
            multiple_stars_out=(stars_out >= 2),
            starting_lineup_changed=lineup_changed
        )
    
    def _apply_adjustments(
        self,
        features: Dict,
        home_adj: TeamAdjustment,
        away_adj: TeamAdjustment
    ) -> Dict:
        """Apply adjustments to features dict"""
        # Net margin (home perspective)
        margin_shift = home_adj.total_mean_adj - away_adj.total_mean_adj
        
        # Apply to means (flexible key names)
        if 'home_mu' in features and 'away_mu' in features:
            features['home_mu'] += margin_shift / 2
            features['away_mu'] -= margin_shift / 2
        elif 'mu_home' in features and 'mu_away' in features:
            features['mu_home'] += margin_shift / 2
            features['mu_away'] -= margin_shift / 2
        
        # Variance (add in quadrature)
        total_var = home_adj.total_variance_adj + away_adj.total_variance_adj
        
        if 'home_sd' in features:
            features['home_sd'] = np.sqrt(features['home_sd']**2 + total_var)
        
        if 'away_sd' in features:
            features['away_sd'] = np.sqrt(features['away_sd']**2 + total_var)
        
        return features
    
    def _compute_confidence(self, impacts: List[DetailedPlayerImpact]) -> float:
        """Compute confidence score"""
        score = 100.0
        
        # Penalize few players
        if len(impacts) < 8:
            score -= 25
        
        # Penalize questionable players
        questionable = sum(1 for p in impacts if 0.25 < p.play_prob < 0.75)
        score -= questionable * 10
        
        # Bonus for many confirmed statuses
        confirmed = sum(1 for p in impacts if p.play_prob in [0.0, 1.0])
        score += min(20, confirmed * 2)
        
        return max(0, min(100, score))
    
    def _build_explanation(
        self,
        game: Game,
        home: TeamAdjustment,
        away: TeamAdjustment,
        original: Dict,
        adjusted: Dict
    ) -> Dict:
        """Build comprehensive explanation"""
        return {
            'game_id': game.game_id,
            'league': self.league,
            'timestamp': datetime.utcnow().isoformat(),
            'confidence': min(home.confidence, away.confidence),
            
            'home_team': {
                'team_id': home.team_id,
                'mean_adjustment': home.total_mean_adj,
                'variance_adjustment': home.total_variance_adj,
                'mvp_impact': home.mvp_impact,
                'superstar_impact': home.superstar_impact,
                'star_impact': home.star_impact,
                'multiple_stars_out': home.multiple_stars_out,
                'lineup_disrupted': home.starting_lineup_changed,
                'key_absences': [
                    {
                        'name': p.player_name,
                        'tier': p.tier,
                        'status': p.status,
                        'impact': p.mean_impact
                    }
                    for p in home.players
                    if abs(p.mean_impact) > 1.5
                ]
            },
            
            'away_team': {
                'team_id': away.team_id,
                'mean_adjustment': away.total_mean_adj,
                'variance_adjustment': away.total_variance_adj,
                'mvp_impact': away.mvp_impact,
                'superstar_impact': away.superstar_impact,
                'star_impact': away.star_impact,
                'multiple_stars_out': away.multiple_stars_out,
                'lineup_disrupted': away.starting_lineup_changed,
                'key_absences': [
                    {
                        'name': p.player_name,
                        'tier': p.tier,
                        'status': p.status,
                        'impact': p.mean_impact
                    }
                    for p in away.players
                    if abs(p.mean_impact) > 1.5
                ]
            },
            
            'net_impact': {
                'margin_shift': home.total_mean_adj - away.total_mean_adj,
                'uncertainty_added': home.total_variance_adj + away.total_variance_adj,
                'original_margin': original.get('home_mu', 0) - original.get('away_mu', 0),
                'adjusted_margin': adjusted.get('home_mu', 0) - adjusted.get('away_mu', 0)
            },
            
            'recommendation': self._generate_recommendation(home, away)
        }
    
    def _generate_recommendation(
        self,
        home: TeamAdjustment,
        away: TeamAdjustment
    ) -> str:
        """Generate betting recommendation"""
        min_conf = min(home.confidence, away.confidence)
        
        if min_conf < self.CONFIDENCE_MINIMUM:
            return "SKIP: Data quality insufficient"
        
        if min_conf < self.CONFIDENCE_RECOMMEND:
            return "CAUTION: Reduce stake 50% due to uncertainty"
        
        if home.multiple_stars_out or away.multiple_stars_out:
            return "ALERT: Multiple star players out - major lineup impact"
        
        return "PROCEED: Good data quality, elite adjustments applied"
    
    def _compute_scenarios(
        self,
        home: TeamAdjustment,
        away: TeamAdjustment
    ) -> Dict:
        """Compute best/worst case scenarios"""
        home_q = [p for p in home.players if 0.25 < p.play_prob < 0.75]
        away_q = [p for p in away.players if 0.25 < p.play_prob < 0.75]
        
        if not home_q and not away_q:
            return {'note': 'No questionable players, single scenario'}
        
        # Best: all play
        best_case = sum(abs(p.mean_impact) for p in home_q + away_q)
        
        # Worst: all sit
        worst_case = -best_case
        
        range_pts = abs(best_case - worst_case)
        
        return {
            'best_case_shift': best_case,
            'worst_case_shift': worst_case,
            'total_range': range_pts,
            'uncertainty_level': (
                'EXTREME' if range_pts > 15 else
                'HIGH' if range_pts > 8 else
                'MODERATE'
            )
        }
    
    def _empty_adjustment(self, team_id: str) -> TeamAdjustment:
        """Empty adjustment"""
        return TeamAdjustment(
            team_id=team_id,
            players=[],
            total_mean_adj=0.0,
            total_variance_adj=0.0,
            mvp_impact=0.0,
            superstar_impact=0.0,
            star_impact=0.0,
            confidence=0.0,
            multiple_stars_out=False,
            starting_lineup_changed=False
        )


# Main convenience function
def apply_elite_adjustments(
    matchup_features: Dict,
    game_id: str,
    league: str = 'nba',
    **kwargs
) -> Tuple[Dict, Dict]:
    """
    **USE THIS FUNCTION** - Main integration point.
    
    Example:
        features = {'home_mu': 105, 'away_mu': 103, 'home_sd': 12, 'away_sd': 12}
        adj_features, explanation = apply_elite_adjustments(features, game_id, 'nba')
        # Continue with your simulation using adj_features
    """
    adjuster = EliteSimulationAdjuster(league)
    return adjuster.apply_elite_adjustments(matchup_features, game_id, **kwargs)
