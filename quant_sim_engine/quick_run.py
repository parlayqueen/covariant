import numpy as np

from quant_sim_engine.sim.covariance_engine import create_covariance_matrix, PlayerCovarianceProfile
from quant_sim_engine.sim.joint_sampler import PlayerDistribution, sample_correlated_players

# -----------------------------
# 1) Player profiles (for covariance)
# -----------------------------
profiles = [
    PlayerCovarianceProfile("A", usage_rate=0.30, assist_rate=0.22, rebound_rate=0.10, position="G", minutes_avg=34.0),
    PlayerCovarianceProfile("B", usage_rate=0.26, assist_rate=0.18, rebound_rate=0.12, position="G", minutes_avg=32.0),
    PlayerCovarianceProfile("C", usage_rate=0.22, assist_rate=0.12, rebound_rate=0.18, position="F", minutes_avg=30.0),
    PlayerCovarianceProfile("D", usage_rate=0.18, assist_rate=0.08, rebound_rate=0.22, position="C", minutes_avg=28.0),
]

cov = create_covariance_matrix(profiles, stat_type="points", league="nba")
print("\n=== Covariance Matrix ===")
print(cov)

# -----------------------------
# 2) Distributions (for sampling)
# -----------------------------
players = [
    PlayerDistribution("A", mean=22.0, std=5.0),
    PlayerDistribution("B", mean=18.0, std=4.5),
    PlayerDistribution("C", mean=15.0, std=4.0),
    PlayerDistribution("D", mean=12.0, std=3.5),
]
player_ids = [p.player_id for p in players]

# -----------------------------
# 3) Joint sampling
# -----------------------------
n_sims = 5000
samples = sample_correlated_players(players, cov, n_sims=n_sims, method="normal")
# samples shape: (n_sims, n_players)

print("\n=== Sample Summary ===")
for j, pid in enumerate(player_ids):
    vals = samples[:, j]
    print(f"{pid}: mean={vals.mean():.2f} std={vals.std(ddof=1):.2f} p90={np.quantile(vals, 0.90):.2f}")

print("\nSimulation complete.")
