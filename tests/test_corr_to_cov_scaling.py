import numpy as np
from quant_sim_engine.sim.covariance_engine import PlayerCovarianceProfile, create_covariance_matrix
from quant_sim_engine.sim.joint_sampler import PlayerDistribution, sample_correlated_players

profiles = [
    PlayerCovarianceProfile("A", 0.28, 0.22, 0.10, "G", 34),
    PlayerCovarianceProfile("B", 0.24, 0.18, 0.08, "G", 33),
    PlayerCovarianceProfile("C", 0.20, 0.12, 0.14, "F", 31),
    PlayerCovarianceProfile("D", 0.16, 0.08, 0.18, "C", 28),
]
corr = create_covariance_matrix(profiles, stat_type="points", league="nba")

players = [
    PlayerDistribution("A", mean=22.0, std=6.0),
    PlayerDistribution("B", mean=18.0, std=5.0),
    PlayerDistribution("C", mean=15.0, std=4.0),
    PlayerDistribution("D", mean=12.0, std=3.0),
]

samples = np.asarray(sample_correlated_players(players, corr, n_sims=40000, method="normal"))
s = samples.std(axis=0)

expected = np.array([6.0, 5.0, 4.0, 3.0])
err = np.max(np.abs(s - expected))

print("sample stds:", np.round(s, 2))
print("max abs error:", float(err))
assert err < 0.25, f"Std scaling drifted too far (max err {err})"
print("OK corr->cov scaling within tolerance")
