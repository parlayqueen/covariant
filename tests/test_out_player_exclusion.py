import glob
import numpy as np
from pathlib import Path

def latest_demo(outdir: str) -> Path:
    demos = sorted(glob.glob(f"{outdir}/demo_*"))
    assert demos, f"No demo runs found in {outdir}"
    return Path(demos[-1])

demo = latest_demo("runs/covariant_usage_shift_audit3")

base_cov = np.load(demo/"baseline"/"cov.npy")
out_cov  = np.load(demo/"star_A_out"/"cov.npy")

print("Demo:", demo)
print("baseline cov shape:", base_cov.shape)
print("star_A_out cov shape:", out_cov.shape)
assert out_cov.shape[0] == base_cov.shape[0] - 1, "Covariance did not shrink by 1 for OUT player."

hdr = (demo/"star_A_out"/"samples.csv").read_text(encoding="utf-8").splitlines()[0].strip()
print("samples.csv header:", hdr)
assert "A" not in hdr.split(","), "OUT player 'A' still present in samples header."

print("OK OUT player excluded from cov + samples")
