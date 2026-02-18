import numpy as np

base = np.load("runs/covariant_smoke/demo_20260217_192950/baseline/cov.npy")
stress = np.load("runs/covariant_smoke/demo_20260217_192950/corr_stress_1_5x/cov.npy")

base_off = np.mean(np.abs(base - np.diag(np.diag(base))))
stress_off = np.mean(np.abs(stress - np.diag(np.diag(stress))))
mult = stress_off / (base_off + 1e-12)

print("multiplier:", mult)
assert 1.45 <= mult <= 1.55, f"Expected ~1.5x, got {mult}"
print("OK stress multiplier within tolerance")
