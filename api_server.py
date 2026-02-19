#!/usr/bin/env python3
from __future__ import annotations

import os, json, glob, time, subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Covariant Picks API", version="0.1.0")

# Allow your local frontend dev server to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _latest(patterns: List[str]) -> Optional[str]:
    cands: List[str] = []
    for p in patterns:
        cands += glob.glob(str(RUNS_DIR / p))
    cands = sorted(cands)
    return cands[-1] if cands else None

def _read_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "runs_dir": str(RUNS_DIR),
        "time": int(time.time()),
        "odds_api_key_present": bool(os.environ.get("ODDS_API_KEY", "").strip()),
    }

@app.get("/api/runs")
def list_runs():
    files = sorted(glob.glob(str(RUNS_DIR / "*.json")))
    return {"count": len(files), "files": [Path(f).name for f in files][-200:]}

@app.get("/api/runs/latest")
def latest_run():
    path = _latest(["picks_full_*.json", "picks_*.json"])
    if not path:
        raise HTTPException(status_code=404, detail="No runs found.")
    return {"file": Path(path).name, "data": _read_json(path)}

@app.get("/api/runs/{filename}")
def get_run(filename: str):
    path = RUNS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run file not found.")
    return {"file": filename, "data": _read_json(str(path))}

@app.post("/api/generate")
def generate():
    """
    Runs your generator and returns the latest ledger.
    Uses the existing file-writing behavior in generate_picks_full.py.
    """
    cmd = ["python", "generate_picks_full.py"]
    proc = subprocess.run(cmd, cwd=str(Path.cwd()), capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Generate failed:\n{proc.stderr}\n{proc.stdout}")

    path = _latest(["picks_full_*.json", "picks_*.json"])
    if not path:
        raise HTTPException(status_code=500, detail="Generator ran but no ledger file found in runs/")
    return {
        "ok": True,
        "stdout": proc.stdout[-4000:],  # keep response size sane
        "file": Path(path).name,
        "data": _read_json(path),
    }

@app.post("/api/grade")
def grade(ledger: str = ""):
    """
    Grades outcomes using grade_outcomes.py.
    Requires ODDS_API_KEY env var and plan access to scores endpoint.
    """
    cmd = ["python", "grade_outcomes.py"]
    if ledger:
        cmd += ["--ledger", ledger]
    cmd += ["--csv", "auto"]

    proc = subprocess.run(cmd, cwd=str(Path.cwd()), capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Grade failed:\n{proc.stderr}\n{proc.stdout}")

    # Return newest graded file if present
    graded = _latest(["graded_*.json"])
    payload = {"ok": True, "stdout": proc.stdout[-4000:]}
    if graded:
        payload["graded_file"] = Path(graded).name
        payload["graded_data"] = _read_json(graded)
    return payload
