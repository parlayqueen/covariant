from quant_sim_engine.sim.covariance_engine import PlayerCovarianceProfile, create_covariance_matrix
from quant_sim_engine.sim.joint_sampler import PlayerDistribution, sample_correlated_players

# Correct schema:
# (player_id, usage_rate, assist_rate, rebound_rate, position, minutes_avg)

players_cov = [
    PlayerCovarianceProfile("A", 0.30, 0.22, 0.10, "G", 34),
    PlayerCovarianceProfile("B", 0.25, 0.18, 0.12, "G", 32),
]

players_dist = [
    PlayerDistribution("A", 25, 6),
    PlayerDistribution("B", 20, 5),
]

cov = create_covariance_matrix(players_cov, "points")
samples = sample_correlated_players(players_dist, cov, 5000)

print("SUCCESS:", samples.shape)
