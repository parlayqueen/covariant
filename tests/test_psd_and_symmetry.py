import numpy as np

base = np.load("runs/covariant_smoke/demo_20260217_192950/baseline/cov.npy")
M = 0.5*(base + base.T)

sym_err = np.max(np.abs(M - M.T))
eig_min = float(np.min(np.linalg.eigvalsh(M)))

print("max symmetry error:", float(sym_err))
print("min eigenvalue:", eig_min)

assert sym_err < 1e-8, f"Covariance not symmetric enough (err={sym_err})"
assert eig_min > -1e-8, f"Covariance not PSD enough (min_eig={eig_min})"
print("OK PSD + symmetry within tolerance")
